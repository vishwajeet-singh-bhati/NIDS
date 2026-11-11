package nids.core

/** Runtime constants shared with the Python pipeline (scripts/export_golden.py writes runtime_config.txt). */
const val PROTO_TCP = 0
const val PROTO_UDP = 1
const val PROTO_DNS = 2
const val PROTO_OTHER = 3

const val TCP_FIN = 0x01
const val TCP_SYN = 0x02
const val TCP_RST = 0x04
const val TCP_PSH = 0x08
const val TCP_ACK = 0x10

const val N_META = 12
const val N_AGG = 22

/** Fixed channel order of the per-packet metadata tensor (must equal src/utils/config.py META_CHANNELS). */
val META_CHANNELS = listOf(
    "direction", "payload_len", "log_iat", "proto_tcp", "proto_udp", "proto_dns", "proto_other",
    "flag_syn", "flag_fin", "flag_rst", "flag_psh", "flag_ack",
)

class RuntimeConfig(private val m: Map<String, String>) {
    private fun s(k: String) = m[k] ?: error("runtime_config.txt: missing key '$k'")
    fun str(k: String) = s(k)
    fun int(k: String) = s(k).trim().toInt()
    fun dbl(k: String) = s(k).trim().toDouble()
    fun ints(k: String) = s(k).split(",").filter { it.isNotBlank() }.map { it.trim().toInt() }
    fun dbls(k: String) = s(k).trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.map { it.toDouble() }
    fun strs(k: String) = s(k).split(",").map { it.trim() }

    val window get() = int("window")
    val stride get() = int("stride")
    val payloadBytes get() = int("payload_bytes")

    companion object {
        fun parse(text: String): RuntimeConfig {
            val m = HashMap<String, String>()
            for (line in text.lineSequence()) {
                val i = line.indexOf('=')
                if (i > 0) m[line.substring(0, i).trim()] = line.substring(i + 1)
            }
            val c = RuntimeConfig(m)
            require(c.strs("meta_channels") == META_CHANNELS) { "meta channel layout differs from this build" }
            return c
        }
    }
}
