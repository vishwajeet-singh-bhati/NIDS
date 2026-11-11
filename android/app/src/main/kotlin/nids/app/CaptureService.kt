package nids.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import nids.core.Event
import nids.core.PcapStreamReader
import nids.core.RuntimeConfig
import nids.core.StreamEngine
import java.io.File

/** Live state shown by the UI (written from the capture thread, read from the UI thread). */
object Live {
    @Volatile var running = false
    @Volatile var status = "idle"
    @Volatile var packets = 0L
    @Volatile var inferences = 0L
    @Volatile var alerts = 0L
    @Volatile var deviceAlarms = 0L
    @Volatile var activeFlows = 0
    @Volatile var msPerWindow = 0.0
    val recent = ArrayDeque<String>()
}

/**
 * Rooted capture: runs `tcpdump -i any -s 300 -w -` as root, parses the pcap stream, feeds the engine, writes every alert to
 * files/alerts.jsonl. Detection and logging only; nothing is blocked and nothing leaves the phone.
 * Needs a tcpdump binary at TCPDUMP (adb push it to /data/local/tmp and chmod +x).
 */
class CaptureService : Service() {
    @Volatile private var proc: Process? = null
    private var thread: Thread? = null
    private var errText: () -> String = { "" }

    override fun onBind(i: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) { stopSelf(); return START_NOT_STICKY }
        if (Live.running) return START_NOT_STICKY
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel("nids", "NIDS", NotificationManager.IMPORTANCE_LOW))
        startForeground(1, Notification.Builder(this, "nids").setContentTitle("Android NIDS").setContentText("monitoring traffic (log only)")
            .setSmallIcon(android.R.drawable.ic_lock_idle_alarm).build())
        val model = intent?.getStringExtra("model") ?: "model_int8_wonly.tflite"
        Live.running = true
        thread = Thread({ loop(model) }, "nids-capture").also { it.start() }
        return START_NOT_STICKY
    }

    private fun loop(model: String) {
        val log = File(filesDir, "alerts.jsonl")
        try {
            val rc = RuntimeConfig.parse(assets.open("runtime_config.txt").bufferedReader().readText())
            TfliteScorer(this, model).use { scorer ->
                val eng = StreamEngine(rc, scorer, batchSize = 1)
                Live.status = "starting tcpdump as root ..."
                val p = ProcessBuilder("su", "-c", "$TCPDUMP -i any -U -s 300 -w -").start()
                proc = p
                val err = StringBuffer()
                Thread { try { p.errorStream.bufferedReader().forEachLine { if (err.length < 1500) err.append(it).append(' ') } } catch (_: Exception) {} }.start()
                errText = { err.toString().trim() }
                val rd = PcapStreamReader(p.inputStream.buffered(1 shl 16))
                Live.status = "capturing (linktype ${rd.linktype}), model $model"
                var n = 0
                while (Live.running) {
                    val fr = rd.next() ?: break
                    val evs = eng.feedRaw(fr.data, fr.linktype, fr.ts)
                    if (evs.isNotEmpty()) record(log, evs)
                    if (++n % 200 == 0) publish(eng)
                }
                publish(eng)
                Live.status = if (Live.running) "capture ended (tcpdump exited; is root granted and tcpdump at $TCPDUMP?)" else "stopped"
            }
        } catch (e: Exception) {
            val code = try { proc?.let { if (it.waitFor(1, java.util.concurrent.TimeUnit.SECONDS)) "exit ${it.exitValue()}" else "still running" } } catch (_: Exception) { null }
            Live.status = "error: ${e.javaClass.simpleName}: ${e.message}\ncapture process: $code\nstderr: ${errText()}"
        } finally {
            Live.running = false
            proc?.destroy()
            stopForeground(STOP_FOREGROUND_REMOVE)
        }
    }

    private fun record(log: File, evs: List<Event>) {
        log.appendText(evs.joinToString("") { it.toJson() + "\n" })
        synchronized(Live.recent) {
            for (e in evs) { Live.recent.addFirst(e.toJson()); while (Live.recent.size > 30) Live.recent.removeLast() }
        }
    }

    private fun publish(e: StreamEngine) {
        Live.packets = e.packets; Live.inferences = e.inferences; Live.alerts = e.alerts
        Live.deviceAlarms = e.deviceAlarms; Live.activeFlows = e.activeFlows
        Live.msPerWindow = if (e.inferences > 0) e.inferenceNanos / 1e6 / e.inferences else 0.0
    }

    override fun onDestroy() {
        Live.running = false
        proc?.destroy()
        super.onDestroy()
    }

    companion object {
        const val ACTION_STOP = "nids.app.STOP"
        const val TCPDUMP = "/data/local/tmp/tcpdump"
    }
}
