package nids.core

import java.io.InputStream

class Frame(val ts: Double, val linktype: Int, val wirelen: Int, val data: ByteArray)

/** Reads a classic libpcap stream (what `tcpdump -w -` writes), e.g. from a process's stdout. */
class PcapStreamReader(private val inp: InputStream) {
    private var littleEndian = true
    private var nano = false
    var linktype = 0; private set

    init {
        val h = ByteArray(24)
        readFully(h)
        val magic = ((h[0].toInt() and 0xFF) shl 24) or ((h[1].toInt() and 0xFF) shl 16) or ((h[2].toInt() and 0xFF) shl 8) or (h[3].toInt() and 0xFF)
        when (magic) {
            0xd4c3b2a1.toInt() -> { littleEndian = true; nano = false }
            0xa1b2c3d4.toInt() -> { littleEndian = false; nano = false }
            0x4d3cb2a1 -> { littleEndian = true; nano = true }
            0xa1b23c4d.toInt() -> { littleEndian = false; nano = true }
            else -> error("not a classic pcap stream (magic %08x)".format(magic))
        }
        linktype = u32(h, 20).toInt()
    }

    private fun u32(b: ByteArray, o: Int): Long {
        val x = if (littleEndian)
            (b[o].toLong() and 0xFF) or ((b[o + 1].toLong() and 0xFF) shl 8) or ((b[o + 2].toLong() and 0xFF) shl 16) or ((b[o + 3].toLong() and 0xFF) shl 24)
        else
            (b[o + 3].toLong() and 0xFF) or ((b[o + 2].toLong() and 0xFF) shl 8) or ((b[o + 1].toLong() and 0xFF) shl 16) or ((b[o].toLong() and 0xFF) shl 24)
        return x
    }

    private fun readFully(b: ByteArray): Boolean {
        var n = 0
        while (n < b.size) {
            val r = inp.read(b, n, b.size - n)
            if (r < 0) { if (n == 0) return false; error("truncated pcap stream") }
            n += r
        }
        return true
    }

    /** Next frame, or null at end of stream. */
    fun next(): Frame? {
        val h = ByteArray(16)
        if (!readFully(h)) return null
        val sec = u32(h, 0)
        val frac = u32(h, 4)
        val incl = u32(h, 8).toInt()
        val orig = u32(h, 12).toInt()
        require(incl in 0..262144) { "bad record length $incl" }
        val data = ByteArray(incl)
        if (incl > 0 && !readFully(data)) return null
        return Frame(sec + frac / (if (nano) 1e9 else 1e6), linktype, orig, data)
    }
}
