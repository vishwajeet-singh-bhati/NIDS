package nids.app

import android.content.Context
import nids.core.RuntimeConfig
import nids.core.StreamEngine
import kotlin.math.abs

/**
 * On-phone parity test. Replays assets/golden/frames.txt (raw frames) through the SAME Kotlin pipeline with the TFLite model on
 * this phone and compares the logits with the PyTorch reference logits in assets/golden/windows.txt.
 * (The Kotlin pipeline itself was already checked against Python on a PC with the recorded logits: `gradle :core:run`.)
 */
object SelfTest {
    fun run(ctx: Context, model: String): String {
        val rc = RuntimeConfig.parse(ctx.assets.open("runtime_config.txt").bufferedReader().readText())
        val thr = rc.dbl("threshold_logit")
        val expected = ctx.assets.open("golden/windows.txt").bufferedReader().useLines { ls ->
            ls.filter { it.startsWith("W ") }.map { it.split(" ")[4].toDouble() }.toList()
        }
        val got = ArrayList<Double>()
        TfliteScorer(ctx, model).use { sc ->
            val eng = StreamEngine(rc, { items -> val z = sc.score(items); z.forEach { got.add(it) }; z }, batchSize = 1)
            var alerts = 0
            var frames = 0
            val t0 = System.nanoTime()
            ctx.assets.open("golden/frames.txt").bufferedReader().useLines { ls ->
                for (line in ls) {
                    if (line == "END") { alerts += eng.finish().count { it.type == "flow_alert" }; eng.resetCapture(); continue }
                    val p = line.split(" ")
                    val b = ByteArray(p[3].length / 2) { p[3].substring(2 * it, 2 * it + 2).toInt(16).toByte() }
                    alerts += eng.feedRaw(b, p[1].toInt(), p[0].toDouble()).count { it.type == "flow_alert" }
                    frames++
                }
            }
            alerts += eng.finish().count { it.type == "flow_alert" }
            val secs = (System.nanoTime() - t0) / 1e9
            val n = minOf(got.size, expected.size)
            var maxD = 0.0; var sumD = 0.0; var flips = 0
            for (i in 0 until n) {
                val d = abs(got[i] - expected[i]); maxD = maxOf(maxD, d); sumD += d
                if ((got[i] >= thr) != (expected[i] >= thr)) flips++
            }
            return "model $model\nframes $frames, windows scored on phone ${got.size} (Python: ${expected.size})\n" +
                "logit diff vs PyTorch: mean ${"%.4f".format(java.util.Locale.ROOT, sumD / maxOf(n, 1))}, max ${"%.4f".format(java.util.Locale.ROOT, maxD)}\n" +
                "alert-decision flips at threshold ${"%.3f".format(java.util.Locale.ROOT, thr)}: $flips of $n\n" +
                "flow alerts on phone: $alerts\n" +
                "whole replay: ${"%.2f".format(java.util.Locale.ROOT, secs)} s (${"%.2f".format(java.util.Locale.ROOT, eng.inferenceNanos / 1e6 / maxOf(eng.inferences, 1))} ms per window incl. input copy)\n" +
                (if (got.size != expected.size) "WARNING: window count differs from Python\n" else "")
        }
    }
}
