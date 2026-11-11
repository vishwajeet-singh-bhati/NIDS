package nids.core

import java.util.Locale

/**
 * Builds the structured "evidence pack" that is handed to the language model which writes the user-facing alert.
 * Every field is a measured quantity from this detector (nothing is inferred or guessed), and the pack carries its own
 * limitations so the writer cannot present a weak-label detection as a confirmed infection.
 * Call [onEvent] with every Event the engine returns; it returns the JSON pack when the event is a device alarm.
 */
class EvidenceCollector(
    private val rc: RuntimeConfig, private val modelName: String = rc.str("model"), private val maxFlows: Int = 8,
) {
    private val windowS = rc.dbl("device_window_s")
    private val minAlerts = rc.int("device_min_alerts")
    private val threshold = rc.dbl("threshold_logit")
    private val recent = ArrayDeque<Event>()

    private fun q(s: String) = "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
    private fun f(x: Double, d: Int = 3) = String.format(Locale.ROOT, "%.${d}f", x)
    private fun proto(p: Int) = when (p) { 6 -> "tcp"; 17 -> "udp"; 1, 58 -> "icmp"; else -> "other($p)" }

    fun onEvent(ev: Event): String? {
        if (ev.type == "flow_alert") {
            if (recent.isNotEmpty() && ev.ts < recent.last().ts) recent.clear()   // a new, independent capture
            recent.addLast(ev)
            while (recent.isNotEmpty() && ev.ts - recent.first().ts > windowS) recent.removeFirst()
            return null
        }
        if (ev.type != "device_alarm") return null
        val inWin = recent.filter { ev.ts - it.ts in 0.0..windowS }
        if (inWin.isEmpty()) return null
        val dests = inWin.groupingBy { it.responder }.eachCount().entries.sortedByDescending { it.value }
        val servers = inWin.map { it.responder.substringBeforeLast(':') }.toSet()
        val protos = inWin.groupingBy { proto(it.ipProto) }.eachCount()
        val reasons = inWin.groupingBy { it.reason }.eachCount()
        val logits = inWin.map { it.windowLogit }
        val firstTs = inWin.minOf { it.ts }
        val lastTs = inWin.maxOf { it.ts }
        val flows = inWin.takeLast(maxFlows).joinToString(",") {
            """{"seconds_before_alarm":${f(ev.ts - it.ts, 1)},"destination":${q(it.responder)},"protocol":${q(proto(it.ipProto))},""" +
                """"reason":${q(it.reason)},"window_logit":${f(it.windowLogit, 2)},"packets_seen_when_flagged":${it.packetsSeen}}"""
        }
        return buildString {
            append("""{"pack_version":1,"event":"device_alarm","alarm_time_unix":${f(ev.ts, 3)},""")
            append(""""rule":{"min_flow_alerts":$minAlerts,"within_seconds":${f(windowS, 0)}},""")
            append(""""evidence":{"flow_alerts_counted_by_rule":${ev.alertsInWindow},"distinct_flow_alerts_logged":${inWin.size},"span_seconds":${f(lastTs - firstTs, 1)},""")
            append(""""distinct_servers":${servers.size},"protocol_mix":{${protos.entries.joinToString(",") { "${q(it.key)}:${it.value}" }}},""")
            append(""""alert_reasons":{${reasons.entries.joinToString(",") { "${q(it.key)}:${it.value}" }}},""")
            append(""""top_destinations":[${dests.take(5).joinToString(",") { """{"destination":${q(it.key)},"flow_alerts":${it.value}}""" }}],""")
            append(""""window_logit":{"min":${f(logits.min(), 2)},"max":${f(logits.max(), 2)},"alert_threshold":${f(threshold, 3)}},""")
            append(""""recent_flows":[$flows]},""")
            append(""""malware_type":null,"malware_type_note":"this detector does not identify the type of malware","model":${q(modelName)},""")
            append(""""limitations":["labels are weak (capture-level), so this means traffic LOOKS LIKE malware-associated traffic, not that an attack was confirmed",""")
            append(""""the model was trained on 2017 lab captures and may flag normal apps","no traffic content was decrypted or read, and nothing was blocked"]}""")
        }
    }
}
