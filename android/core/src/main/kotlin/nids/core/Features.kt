package nids.core

import kotlin.math.ln1p
import kotlin.math.max
import kotlin.math.sqrt

/** Model inputs for ONE window, in the TFLite (channels-first) layout: meta [12,W], mask [W], payload [B,W] int, agg [22]. */
class WindowTensors(val meta: FloatArray, val mask: FloatArray, val payload: IntArray, val agg: FloatArray)

/** Port of encode_window (feature spec v1.1) + Normalizer + aggregate_features + tensors_from_arrays. */
class WindowEncoder(private val c: RuntimeConfig) {
    private val W = c.window
    private val B = c.payloadBytes
    private val iatMs = c.int("iat_log_unit_ms") != 0
    private val mTr = c.strs("meta_transform")
    private val mMean = c.dbls("meta_mean")
    private val mStd = c.dbls("meta_std")
    private val aTr = c.strs("agg_transform")
    private val aMean = c.dbls("agg_mean")
    private val aStd = c.dbls("agg_std")

    private val CH = META_CHANNELS.withIndex().associate { it.value to it.index }
    private val protoCh = intArrayOf(CH["proto_tcp"]!!, CH["proto_udp"]!!, CH["proto_dns"]!!, CH["proto_other"]!!)

    fun encode(pk: List<PacketRecord>): WindowTensors {
        require(pk.isNotEmpty())
        val n = minOf(pk.size, W)
        val raw = Array(W) { FloatArray(N_META) }
        val mask = FloatArray(W)
        val pay = Array(W) { ByteArray(0) }
        val t0 = pk[0].ts
        var clock = 0.0
        for (i in 0 until n) {
            val p = pk[i]
            val row = raw[i]
            row[CH["direction"]!!] = p.direction.toFloat()
            row[CH["payload_len"]!!] = p.payloadLen.toFloat()
            val newClock = if (i > 0) max(clock, p.ts - t0) else 0.0
            val iat = newClock - clock
            clock = newClock
            row[CH["log_iat"]!!] = ln1p(if (iatMs) iat * 1000.0 else iat).toFloat()
            row[protoCh[p.category]] = 1f
            if (p.tcpFlags != 0) {
                val f = p.tcpFlags
                row[CH["flag_syn"]!!] = if ((f and TCP_SYN) != 0) 1f else 0f
                row[CH["flag_fin"]!!] = if ((f and TCP_FIN) != 0) 1f else 0f
                row[CH["flag_rst"]!!] = if ((f and TCP_RST) != 0) 1f else 0f
                row[CH["flag_psh"]!!] = if ((f and TCP_PSH) != 0) 1f else 0f
                row[CH["flag_ack"]!!] = if ((f and TCP_ACK) != 0) 1f else 0f
            }
            mask[i] = 1f
            pay[i] = if (p.payload.size > B) p.payload.copyOf(B) else p.payload
        }
        // ---- normalised meta, channels-first, padding rows zero
        val meta = FloatArray(N_META * W)
        for (i in 0 until W) {
            if (mask[i] == 0f) continue
            for (ch in 0 until N_META) {
                var x = raw[i][ch].toDouble()
                if (mTr[ch] == "log1p_zscore") x = ln1p(x)
                if (mTr[ch] != "none") x = (x - mMean[ch]) / mStd[ch]
                meta[ch * W + i] = x.toFloat()
            }
        }
        // ---- payload tokens (byte+1, 0 = padding), channels-first
        val payload = IntArray(B * W)
        for (i in 0 until n) {
            val pb = pay[i]
            for (j in pb.indices) payload[j * W + i] = (pb[j].toInt() and 0xFF) + 1
        }
        return WindowTensors(meta, mask, payload, aggregate(raw, mask))
    }

    private fun aggregate(raw: Array<FloatArray>, mask: FloatArray): FloatArray {
        val d = { i: Int -> raw[i][CH["direction"]!!].toDouble() }
        val pl = { i: Int -> raw[i][CH["payload_len"]!!].toDouble() }
        val iat = { i: Int -> raw[i][CH["log_iat"]!!].toDouble() }
        var sm = 0.0
        for (i in 0 until W) sm += mask[i]
        val n = max(sm, 1.0)
        fun wsum(f: (Int) -> Double, w: (Int) -> Double): Double { var s = 0.0; for (i in 0 until W) s += f(i) * w(i); return s }
        val m = { i: Int -> mask[i].toDouble() }
        val fwd = { i: Int -> if (d(i) > 0) mask[i].toDouble() else 0.0 }
        val rev = { i: Int -> if (d(i) < 0) mask[i].toDouble() else 0.0 }
        val nFwd = wsum({ 1.0 }, fwd)
        val nRev = wsum({ 1.0 }, rev)
        fun mmean(f: (Int) -> Double, w: (Int) -> Double) = wsum(f, w) / max(wsum({ 1.0 }, w), 1.0)
        val meanPl = mmean(pl, m)
        val stdPl = sqrt(max(mmean({ pl(it) * pl(it) }, m) - meanPl * meanPl, 0.0))
        val meanIat = mmean(iat, m)
        val stdIat = sqrt(max(mmean({ iat(it) * iat(it) }, m) - meanIat * meanIat, 0.0))
        var maxPl = 0.0
        var maxIat = 0.0
        var zeros = 0.0
        for (i in 0 until W) { maxPl = max(maxPl, pl(i) * mask[i]); maxIat = max(maxIat, iat(i) * mask[i]); if (pl(i) == 0.0) zeros += mask[i] }
        fun frac(ch: String) = wsum({ raw[it][CH[ch]!!].toDouble() }, m) / n
        fun count(ch: String) = wsum({ raw[it][CH[ch]!!].toDouble() }, m)
        val cols = doubleArrayOf(
            sm, nFwd / n, nFwd, nRev,
            wsum(pl, fwd), wsum(pl, rev), mmean(pl, fwd), mmean(pl, rev),
            maxPl, stdPl, zeros / n,
            meanIat, maxIat, stdIat,
            frac("proto_tcp"), frac("proto_udp"), frac("proto_dns"), frac("proto_other"),
            count("flag_syn"), count("flag_fin"), count("flag_rst"), count("flag_psh"),
        )
        val out = FloatArray(N_AGG)
        for (k in 0 until N_AGG) {
            var x = cols[k]
            if (aTr[k] == "log1p") x = ln1p(x)
            out[k] = ((x - aMean[k]) / aStd[k]).toFloat()
        }
        return out
    }
}
