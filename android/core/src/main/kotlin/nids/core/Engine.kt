package nids.core

import java.util.Locale
import kotlin.math.exp

class WindowKey(val flowIndex: Int, val start: Int, val k: Int) {
    override fun toString() = "flow=$flowIndex start=$start k=$k"
}

class PendingWindow(val key: WindowKey, val tensors: WindowTensors)

/** Anything that turns windows into logits: TFLite on the phone, recorded logits in the parity test. */
fun interface Scorer { fun score(items: List<PendingWindow>): DoubleArray }

class Event(
    val type: String,            // "flow_alert" | "device_alarm"
    val ts: Double,
    val flowIndex: Int = -1,
    val initiator: String = "", val responder: String = "", val ipProto: Int = 0,
    val reason: String = "",
    val windowLogit: Double = 0.0, val flowScore: Double = 0.0,
    val scoredWindows: Int = 0, val packetsSeen: Int = 0, val decidedOnPrefixK: Int = 0,
    val flowStartTs: Double = 0.0,
    val rule: String = "",
) {
    fun toJson(): String = if (type == "device_alarm")
        """{"event":"device_alarm","ts":${"%.3f".format(Locale.ROOT, ts)},"rule":"$rule","note":"detection/log only, nothing is blocked"}"""
    else
        """{"event":"flow_alert","ts":${"%.3f".format(Locale.ROOT, ts)},"flow_index":$flowIndex,"initiator":"$initiator","responder":"$responder",""" +
            """"ip_proto":$ipProto,"reason":"$reason","window_logit":${"%.4f".format(Locale.ROOT, windowLogit)},"flow_score":${"%.4f".format(Locale.ROOT, flowScore)},""" +
            """"window_prob":${"%.4f".format(Locale.ROOT, 1.0 / (1.0 + exp(-windowLogit)))},"scored_windows":$scoredWindows,"packets_seen":$packetsSeen,""" +
            """"decided_on_prefix_k":$decidedOnPrefixK,"seconds_since_flow_start":${"%.3f".format(Locale.ROOT, ts - flowStartTs)},""" +
            """"note":"malicious-associated traffic pattern (weak-label model); detection/log only, nothing is blocked"}"""
}

class StreamConfig(
    val checkpoints: List<Int>, val finalMinPackets: Int, val threshold: Double, val emaAlpha: Double, val minHits: Int,
    val hardMargin: Double, val maxScoredWindows: Int, val maxActiveFlows: Int, val cooldownS: Double,
    val drainEvery: Int, val deviceWindowS: Double, val deviceMinAlerts: Int, val stride: Int, val window: Int,
    val dnsPorts: Set<Int>, val payloadBytes: Int,
) {
    companion object {
        fun from(c: RuntimeConfig): StreamConfig {
            val w = c.window
            return StreamConfig(
                c.ints("checkpoints").map { minOf(it, w) }.toSortedSet().plus(w).toList().sorted(), c.int("final_min_packets"),
                c.dbl("threshold_logit"), c.dbl("ema_alpha"), c.int("min_hits"), c.dbl("hard_margin"), c.int("max_scored_windows"),
                c.int("max_active_flows"), c.dbl("cooldown_s"), c.int("drain_every"), c.dbl("device_window_s"), c.int("device_min_alerts"),
                c.stride, w, c.ints("dns_ports").toSet(), c.payloadBytes,
            )
        }
    }
}

private class FlowState {
    val pol = PolicyState()
    val scoredAt = HashSet<Long>()
}

private class Pending(val flow: Flow, val start: Int, val k: Int, val tensors: WindowTensors, val ts: Double)

/**
 * Port of src/stream/engine.py StreamDetector: packets -> flows -> windows scored at the checkpoints (8/16/32 packets)
 * -> flow alert policy -> device alarm. Detection and logging only. [batchSize] = 1 scores every window immediately
 * (phone); the policy sees windows in the same FIFO order as the Python engine, so results are identical.
 */
class StreamEngine(
    private val rc: RuntimeConfig, private val scorer: Scorer, private val batchSize: Int = 1,
) {
    private val sc = StreamConfig.from(rc)
    private val flowCfg = FlowConfig.from(rc)
    private val encoder = WindowEncoder(rc)
    private val pol = PolicyParams(sc.threshold, sc.emaAlpha, sc.minHits, sc.hardMargin)
    private var tracker = FlowTracker(flowCfg)
    private var state = HashMap<Int, FlowState>()
    private var queue = ArrayList<Pending>()
    private var lastAlert = HashMap<String, Double>()
    private var sinceDrain = 0
    private var burst = BurstDetector(sc.deviceWindowS, sc.deviceMinAlerts)
    val parseStats = ParseStats()
    var packets = 0L; private set
    var inferences = 0L; private set
    var alerts = 0L; private set
    var deviceAlarms = 0L; private set
    var suppressedByCooldown = 0L; private set
    var inferenceNanos = 0L; private set
    val activeFlows get() = tracker.activeCount

    fun feedRaw(buf: ByteArray, linktype: Int, ts: Double): List<Event> {
        val p = PacketParser.decode(buf, linktype, ts, sc.payloadBytes, sc.dnsPorts, parseStats) ?: return emptyList()
        return feed(p)
    }

    fun feed(pkt: ParsedPacket): List<Event> {
        packets++
        val flow = tracker.process(pkt)
        if (flow != null) afterAppend(flow, pkt.ts)
        sinceDrain++
        val out = ArrayList<Event>()
        if (queue.size >= batchSize) out += runInference()
        if (sinceDrain >= sc.drainEvery) out += housekeeping()
        return out
    }

    private fun afterAppend(flow: Flow, ts: Double) {
        val st = state.getOrPut(flow.flowIndex) { FlowState() }
        val n = flow.packets.size
        for (k in sc.checkpoints) {
            val s = n - k
            if (s < 0 || s % sc.stride != 0) continue
            if (s / sc.stride >= sc.maxScoredWindows) continue
            enqueue(flow, st, s, k, ts)
        }
    }

    private fun enqueue(flow: Flow, st: FlowState, s: Int, k: Int, ts: Double) {
        if (!st.scoredAt.add(s.toLong() * 100_000L + k)) return
        queue.add(Pending(flow, s, k, encoder.encode(flow.packets.subList(s, minOf(s + k, flow.packets.size))), ts))
    }

    private fun finalise(flow: Flow) {
        val st = state[flow.flowIndex] ?: return
        val n = flow.packets.size
        if (n >= 2) {
            val s = ((n - 1) / sc.stride) * sc.stride
            val k = n - s
            if (k >= sc.finalMinPackets && s / sc.stride < sc.maxScoredWindows) enqueue(flow, st, s, k, flow.lastTs)
        }
    }

    private fun runInference(): List<Event> {
        if (queue.isEmpty()) return emptyList()
        val q = queue
        queue = ArrayList()
        val out = ArrayList<Event>()
        var i = 0
        while (i < q.size) {
            val chunk = q.subList(i, minOf(i + maxOf(batchSize, 1), q.size))
            val t0 = System.nanoTime()
            val z = scorer.score(chunk.map { PendingWindow(WindowKey(it.flow.flowIndex, it.start, it.k), it.tensors) })
            inferenceNanos += System.nanoTime() - t0
            inferences += chunk.size
            for (j in chunk.indices) out += applyPolicy(chunk[j], z[j])
            i += maxOf(batchSize, 1)
        }
        return out
    }

    private fun applyPolicy(p: Pending, z: Double): List<Event> {
        val st = state[p.flow.flowIndex] ?: return emptyList()
        val reason = Policy.step(st.pol, z, pol) ?: return emptyList()
        return emit(p, st, z, reason)
    }

    private fun emit(p: Pending, st: FlowState, z: Double, reason: String): List<Event> {
        val f = p.flow
        val out = ArrayList<Event>()
        if (burst.add(p.ts)) {
            deviceAlarms++
            out += Event("device_alarm", p.ts, rule = ">= ${sc.deviceMinAlerts} flow alerts within ${"%g".format(Locale.ROOT, sc.deviceWindowS)}s")
        }
        val dedupe = "${f.initiator.ip}|${f.responder.ip}|${f.responder.port}|${f.ipProto}"
        val last = lastAlert[dedupe]
        if (last != null && p.ts - last < sc.cooldownS) {
            suppressedByCooldown++
        } else {
            lastAlert[dedupe] = p.ts
            alerts++
            out.add(0, Event("flow_alert", p.ts, f.flowIndex, f.initiator.toString(), f.responder.toString(), f.ipProto, reason, z,
                st.pol.ema!!, st.pol.scored, p.start + p.k, p.k, f.startTs))
        }
        return out
    }

    private fun housekeeping(): List<Event> {
        sinceDrain = 0
        if (tracker.activeCount > sc.maxActiveFlows) tracker.evictOldest(sc.maxActiveFlows)
        val done = tracker.drainFinished()
        for (f in done) finalise(f)
        val out = runInference()
        for (f in done) state.remove(f.flowIndex)
        return out
    }

    /** End of input: finalise every flow, score what is pending, forget all flow state. */
    fun finish(): List<Event> {
        for (f in tracker.flush()) finalise(f)
        val out = runInference()
        state.clear()
        return out
    }

    /** Fresh, independent input (the next pcap in a replay): new flow table and no alert history. */
    fun resetCapture() {
        tracker = FlowTracker(flowCfg)
        state = HashMap(); queue = ArrayList(); lastAlert = HashMap(); sinceDrain = 0
        burst = BurstDetector(sc.deviceWindowS, sc.deviceMinAlerts)
    }
}
