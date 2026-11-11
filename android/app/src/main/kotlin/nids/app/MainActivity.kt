package nids.app

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File

class MainActivity : Activity() {
    private lateinit var out: TextView
    private var model = "model_int8_wonly.tflite"
    private val ui = Handler(Looper.getMainLooper())
    private val tick = object : Runnable {
        override fun run() { render(); ui.postDelayed(this, 1000) }
    }

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        if (android.os.Build.VERSION.SDK_INT >= 33) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 24, 24, 24) }
        fun btn(t: String, f: (Button) -> Unit) = Button(this).apply { text = t; setOnClickListener { f(this) } }.also { col.addView(it) }
        btn("Model: INT8 (weights)") { v ->
            model = if (model.contains("int8")) "model_fp32.tflite" else "model_int8_wonly.tflite"
            v.text = if (model.contains("int8")) "Model: INT8 (weights)" else "Model: FP32"
        }
        btn("Self-test (replay golden frames, compare with PyTorch)") {
            out.text = "running self-test ..."
            Thread { val r = try { SelfTest.run(this, model) } catch (e: Exception) { "self-test failed: $e" }; ui.post { out.text = r } }.start()
        }
        btn("Start live capture (root)") {
            startForegroundService(Intent(this, CaptureService::class.java).putExtra("model", model))
        }
        btn("Stop") { startService(Intent(this, CaptureService::class.java).setAction(CaptureService.ACTION_STOP)) }
        out = TextView(this).apply { textSize = 12f; typeface = android.graphics.Typeface.MONOSPACE }
        col.addView(ScrollView(this).apply { addView(out) })
        setContentView(col)
    }

    override fun onResume() { super.onResume(); ui.post(tick) }
    override fun onPause() { ui.removeCallbacks(tick); super.onPause() }

    private fun render() {
        if (!Live.running && Live.packets == 0L && Live.status == "idle") return
        val recent = synchronized(Live.recent) { Live.recent.take(8).joinToString("\n") }
        out.text = "status: ${Live.status}\npackets ${Live.packets} | active flows ${Live.activeFlows} | windows scored ${Live.inferences} " +
            "(${"%.2f".format(java.util.Locale.ROOT, Live.msPerWindow)} ms each)\nflow alerts ${Live.alerts} | DEVICE ALARMS ${Live.deviceAlarms}\n" +
            "log: ${File(filesDir, "alerts.jsonl")}\n" +
            (if (Live.lastPack.isNotEmpty()) "evidence pack ready: files/evidence_packs.jsonl\n" else "") + "\nlast events:\n$recent"
    }
}
