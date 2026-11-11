package nids.core

import java.io.File
import kotlin.math.abs
import kotlin.system.exitProcess

/**
 * Parity check: replays golden/frames.txt through the Kotlin engine and compares with what the Python reference
 * (scripts/export_golden.py) produced: same windows in the same order, same model-input tensors, same alerts.
 * The scorer returns the Python logits, so this isolates parsing / flows / windows / normalisation / policy from the model.
 *   gradle :core:run --args="<golden dir>"
 */
object GoldenCheck {
    class Exp(val flow: Int, val start: Int, val k: Int, val logit: Double) {
        var meta: DoubleArray? = null; var agg: DoubleArray? = null; var pay: IntArray? = null; var mask: DoubleArray? = null
    }

    fun hex(s: String): ByteArray { val b = ByteArray(s.length / 2); for (i in b.indices) b[i] = s.substring(2 * i, 2 * i + 2).toInt(16).toByte(); return b }
    fun dbls(s: String) = s.trim().split(" ").filter { it.isNotEmpty() }.map { it.toDouble() }.toDoubleArray()

    @JvmStatic
    fun main(args: Array<String>) {
        val dir = File(args.getOrElse(0) { "golden" })
        val rc = RuntimeConfig.parse(File(dir, "runtime_config.txt").readText())
        val exp = ArrayList<Exp>()
        for (line in File(dir, "windows.txt").readLines()) {
            val t = line.split(" ", limit = 2)
            when (t[0]) {
                "W" -> { val p = line.split(" "); exp.add(Exp(p[1].toInt(), p[2].toInt(), p[3].toInt(), p[4].toDouble())) }
                "M" -> exp.last().meta = dbls(t[1])
                "A" -> exp.last().agg = dbls(t[1])
                "P" -> exp.last().pay = t[1].trim().split(" ").map { it.toInt() }.toIntArray()
                "MASK" -> exp.last().mask = dbls(t[1])
            }
        }
        val expEvents = File(dir, "events.txt").readLines().filter { it.isNotBlank() }

        var idx = 0
        var bad = 0
        var maxMeta = 0.0; var maxAgg = 0.0; var payMismatch = 0; var tensorChecked = 0
        fun fail(msg: String) { if (bad++ < 15) println("MISMATCH: $msg") }
        val scorer = Scorer { items ->
            DoubleArray(items.size) { j ->
                val it = items[j]
                val e = exp.getOrNull(idx)
                if (e == null) { fail("extra window #$idx ${it.key}"); 0.0 }
                else {
                    if (e.flow != it.key.flowIndex || e.start != it.key.start || e.k != it.key.k)
                        fail("window #$idx expected flow=${e.flow} start=${e.start} k=${e.k}, got ${it.key}")
                    e.meta?.let { m ->
                        tensorChecked++
                        for (q in m.indices) maxMeta = maxOf(maxMeta, abs(m[q] - it.tensors.meta[q]))
                        for (q in e.agg!!.indices) maxAgg = maxOf(maxAgg, abs(e.agg!![q] - it.tensors.agg[q]))
                        for (q in e.pay!!.indices) if (e.pay!![q] != it.tensors.payload[q]) payMismatch++
                        for (q in e.mask!!.indices) if (e.mask!![q].toFloat() != it.tensors.mask[q]) fail("mask differs window #$idx")
                    }
                    idx++
                    e.logit
                }
            }
        }
        val eng = StreamEngine(rc, scorer, batchSize = 1)
        val got = ArrayList<String>()
        fun note(evs: List<Event>) {
            for (ev in evs) got.add(if (ev.type == "device_alarm") "device_alarm %.6f".format(java.util.Locale.ROOT, ev.ts)
            else "flow_alert ${ev.flowIndex} ${ev.reason} %.6f ${ev.windowLogit}".format(java.util.Locale.ROOT, ev.ts))
        }
        var nframes = 0
        for (line in File(dir, "frames.txt").useLines { it.toList() }) {
            if (line == "END") { note(eng.finish()); eng.resetCapture(); continue }
            val p = line.split(" ")
            note(eng.feedRaw(hex(p[3]), p[1].toInt(), p[0].toDouble()))
            nframes++
        }
        note(eng.finish())

        if (idx != exp.size) fail("scored $idx windows, Python scored ${exp.size}")
        // events: compare type/flow/reason exactly, time to 1 ms, logit to 1e-3
        if (got.size != expEvents.size) fail("event count: Kotlin ${got.size} vs Python ${expEvents.size}")
        for (i in 0 until minOf(got.size, expEvents.size)) {
            val a = got[i].split(" "); val b = expEvents[i].split(" ")
            val same = a.size == b.size && a[0] == b[0] && (if (a[0] == "device_alarm") abs(a[1].toDouble() - b[1].toDouble()) < 1e-3
            else a[1] == b[1] && a[2] == b[2] && abs(a[3].toDouble() - b[3].toDouble()) < 1e-3)
            if (!same) fail("event #$i Kotlin '${got[i]}' vs Python '${expEvents[i]}'")
        }
        if (maxMeta > 1e-3) fail("meta max abs diff $maxMeta")
        if (maxAgg > 5e-3) fail("agg max abs diff $maxAgg")
        if (payMismatch > 0) fail("payload tokens differ: $payMismatch values")
        println("frames=$nframes windows=${exp.size} (tensors checked: $tensorChecked) events=${expEvents.size}")
        println("max|meta diff|=%.2e  max|agg diff|=%.2e  payload mismatches=$payMismatch".format(java.util.Locale.ROOT, maxMeta, maxAgg))
        println(if (bad == 0) "PARITY OK" else "PARITY FAILED ($bad problems)")
        exitProcess(if (bad == 0) 0 else 1)
    }
}
