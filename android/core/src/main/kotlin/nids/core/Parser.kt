package nids.core

/** One decoded IP packet. Mirrors src/data/parse_pcap.py ParsedPacket (IP addresses/ports are for flow grouping only). */
class ParsedPacket(
    val ts: Double, val ipProto: Int, val srcIp: String, val dstIp: String, val sport: Int, val dport: Int,
    val category: Int, val tcpFlags: Int, val seq: Long, val payloadLen: Int, val payload: ByteArray,
)

class ParseStats {
    var frames = 0L; var parsed = 0L; var nonIp = 0L; var malformed = 0L; var fragSkipped = 0L; var unsupportedLink = 0L
}

object PacketParser {
    const val DLT_NULL = 0
    const val DLT_EN10MB = 1
    const val DLT_RAW = 101
    const val DLT_RAW_12 = 12
    const val DLT_RAW_14 = 14
    const val DLT_LINUX_SLL = 113
    const val DLT_IPV4 = 228
    const val DLT_IPV6 = 229
    const val DLT_LINUX_SLL2 = 276
    private val SUPPORTED = setOf(DLT_NULL, DLT_EN10MB, DLT_RAW, DLT_RAW_12, DLT_RAW_14, DLT_LINUX_SLL, DLT_IPV4, DLT_IPV6, DLT_LINUX_SLL2)

    private class Malformed : Exception()

    private fun u8(b: ByteArray, i: Int) = b[i].toInt() and 0xFF
    private fun u16(b: ByteArray, i: Int) = (u8(b, i) shl 8) or u8(b, i + 1)
    private fun u32(b: ByteArray, i: Int) = (u16(b, i).toLong() shl 16) or u16(b, i + 2).toLong()

    private fun ip4(b: ByteArray, i: Int) = "${u8(b, i)}.${u8(b, i + 1)}.${u8(b, i + 2)}.${u8(b, i + 3)}"
    private fun ip6(b: ByteArray, i: Int): String {
        val sb = StringBuilder()
        for (g in 0 until 8) {
            if (g > 0) sb.append(':')
            sb.append(Integer.toHexString(u16(b, i + 2 * g)))
        }
        return sb.toString()
    }

    /** (ip version, ip header offset) or null when the frame is not IP. */
    private fun stripLink(buf: ByteArray, lt: Int): IntArray? {
        val n = buf.size
        val proto: Int
        val off: Int
        when (lt) {
            DLT_EN10MB -> {
                if (n < 14) return null
                var o = 12
                var et = u16(buf, o); o += 2
                for (r in 0 until 2) {
                    if (et == 0x8100 || et == 0x88A8 || et == 0x9100) {
                        if (n < o + 4) return null
                        et = u16(buf, o + 2); o += 4
                    }
                }
                return when (et) { 0x0800 -> intArrayOf(4, o); 0x86DD -> intArrayOf(6, o); else -> null }
            }
            DLT_LINUX_SLL -> { if (n < 16) return null; proto = u16(buf, 14); off = 16 }
            DLT_LINUX_SLL2 -> { if (n < 20) return null; proto = u16(buf, 0); off = 20 }
            DLT_NULL -> {
                if (n < 4) return null
                val le = (u8(buf, 3).toLong() shl 24) or (u8(buf, 2).toLong() shl 16) or (u8(buf, 1).toLong() shl 8) or u8(buf, 0).toLong()
                val be = u32(buf, 0)
                val fam = if (le < 256) le else be
                off = 4
                proto = if (fam == 2L) 0x0800 else if (fam == 10L || fam == 23L || fam == 24L || fam == 28L || fam == 30L) 0x86DD else 0
            }
            DLT_RAW, DLT_RAW_12, DLT_RAW_14, DLT_IPV4, DLT_IPV6 -> {
                if (n < 1) return null
                val ver = u8(buf, 0) shr 4
                return if (ver == 4 || ver == 6) intArrayOf(ver, 0) else null
            }
            else -> return null
        }
        return when (proto) { 0x0800 -> intArrayOf(4, off); 0x86DD -> intArrayOf(6, off); else -> null }
    }

    /** Returns null for non-IP, malformed, or non-first fragments (same rules as the Python decoder). */
    fun decode(buf: ByteArray, linktype: Int, ts: Double, payloadBytesKept: Int, dnsPorts: Set<Int>, stats: ParseStats? = null): ParsedPacket? {
        stats?.frames = (stats?.frames ?: 0L) + 1
        if (linktype !in SUPPORTED) { stats?.unsupportedLink = (stats?.unsupportedLink ?: 0L) + 1; return null }
        val loc = stripLink(buf, linktype)
        if (loc == null) { stats?.nonIp = (stats?.nonIp ?: 0L) + 1; return null }
        val ver = loc[0]
        val ipOff = loc[1]
        val caplen = buf.size
        try {
            var declaredEnd: Int
            var l4Off: Int
            val proto: Int
            val src: String
            val dst: String
            if (ver == 4) {
                if (caplen < ipOff + 20) throw Malformed()
                val vihl = u8(buf, ipOff)
                val ihl = (vihl and 0x0F) * 4
                if ((vihl shr 4) != 4 || ihl < 20) throw Malformed()
                val totalLen = u16(buf, ipOff + 2)
                val flOff = u16(buf, ipOff + 6)
                val fragOff = flOff and 0x1FFF
                proto = u8(buf, ipOff + 9)
                src = ip4(buf, ipOff + 12)
                dst = ip4(buf, ipOff + 16)
                declaredEnd = if (totalLen == 0) caplen else if (totalLen < ihl) throw Malformed() else ipOff + totalLen
                l4Off = ipOff + ihl
                if (fragOff > 0) { stats?.fragSkipped = (stats?.fragSkipped ?: 0L) + 1; return null }
            } else {
                if (caplen < ipOff + 40) throw Malformed()
                if ((u8(buf, ipOff) shr 4) != 6) throw Malformed()
                val plen = u16(buf, ipOff + 4)
                var nxt = u8(buf, ipOff + 6)
                src = ip6(buf, ipOff + 8)
                dst = ip6(buf, ipOff + 24)
                declaredEnd = if (plen != 0) ipOff + 40 + plen else caplen
                l4Off = ipOff + 40
                for (r in 0 until 8) {
                    if (nxt == 0 || nxt == 43 || nxt == 60) {
                        if (caplen < l4Off + 2) throw Malformed()
                        val hl = (u8(buf, l4Off + 1) + 1) * 8
                        nxt = u8(buf, l4Off); l4Off += hl
                    } else if (nxt == 51) {
                        if (caplen < l4Off + 2) throw Malformed()
                        val hl = (u8(buf, l4Off + 1) + 2) * 4
                        nxt = u8(buf, l4Off); l4Off += hl
                    } else if (nxt == 44) {
                        if (caplen < l4Off + 8) throw Malformed()
                        val fo = u16(buf, l4Off + 2)
                        if ((fo shr 3) > 0) { stats?.fragSkipped = (stats?.fragSkipped ?: 0L) + 1; return null }
                        nxt = u8(buf, l4Off); l4Off += 8
                    } else break
                }
                proto = nxt
            }
            declaredEnd = maxOf(declaredEnd, l4Off)
            var sport = 0
            var dport = 0
            var flags = 0
            var seq = 0L
            val hdrEnd: Int
            var category: Int
            if (proto == 6) {
                if (caplen < l4Off + 20) throw Malformed()
                sport = u16(buf, l4Off); dport = u16(buf, l4Off + 2); seq = u32(buf, l4Off + 4)
                val offFlags = u16(buf, l4Off + 12)
                val doff = (offFlags shr 12) * 4
                if (doff < 20) throw Malformed()
                flags = offFlags and 0xFF
                hdrEnd = l4Off + doff
                category = PROTO_TCP
            } else if (proto == 17) {
                if (caplen < l4Off + 8) throw Malformed()
                sport = u16(buf, l4Off); dport = u16(buf, l4Off + 2)
                hdrEnd = l4Off + 8
                category = PROTO_UDP
            } else {
                hdrEnd = l4Off
                category = PROTO_OTHER
            }
            if ((category == PROTO_TCP || category == PROTO_UDP) && (sport in dnsPorts || dport in dnsPorts)) category = PROTO_DNS
            val payloadLen = maxOf(0, declaredEnd - hdrEnd)
            val capEnd = minOf(caplen, declaredEnd)
            val keep = minOf(payloadBytesKept, payloadLen, maxOf(0, capEnd - hdrEnd))
            val payload = if (keep > 0) buf.copyOfRange(hdrEnd, hdrEnd + keep) else ByteArray(0)
            if (proto == 6 && hdrEnd > capEnd && declaredEnd <= caplen) throw Malformed()
            stats?.parsed = (stats?.parsed ?: 0L) + 1
            return ParsedPacket(ts, proto, src, dst, sport, dport, category, flags, seq, payloadLen, payload)
        } catch (e: Malformed) {
            stats?.malformed = (stats?.malformed ?: 0L) + 1
            return null
        } catch (e: IndexOutOfBoundsException) {
            stats?.malformed = (stats?.malformed ?: 0L) + 1
            return null
        }
    }
}
