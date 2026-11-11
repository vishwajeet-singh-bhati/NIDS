package nids.core

class Endpoint(val ip: String, val port: Int) : Comparable<Endpoint> {
    override fun compareTo(other: Endpoint): Int { val c = ip.compareTo(other.ip); return if (c != 0) c else port.compareTo(other.port) }
    override fun equals(other: Any?) = other is Endpoint && other.ip == ip && other.port == port
    override fun hashCode() = ip.hashCode() * 31 + port
    override fun toString() = "$ip:$port"
}

data class FlowKey(val proto: Int, val a: Endpoint, val b: Endpoint)

class PacketRecord(
    val ts: Double, val direction: Int, val category: Int, val payloadLen: Int, val payload: ByteArray, val tcpFlags: Int,
)

class Flow(
    val flowIndex: Int, val key: FlowKey, val initiator: Endpoint, val responder: Endpoint, val ipProto: Int,
    val category: Int, val startTs: Double, var lastTs: Double,
) {
    val packets = ArrayList<PacketRecord>()
    var retransmissions = 0
    var packetsDroppedCap = 0
    var synSeen = false
    var synSeq: Long? = null
    val finDir = HashSet<Int>()
    var closed = false
    var closedTs = 0.0
    val maxEndSeq = HashMap<Int, Long>()
}

class FlowConfig(
    val tcpIdleTimeoutS: Double = 120.0, val udpIdleTimeoutS: Double = 60.0, val otherIdleTimeoutS: Double = 60.0,
    val maxFlowDurationS: Double = 600.0, val tcpTimeWaitS: Double = 5.0, val maxPacketsPerFlow: Int = 1500,
    val dropTcpRetransmissions: Boolean = true, val expireCheckIntervalS: Double = 1.0,
) {
    companion object {
        fun from(c: RuntimeConfig) = FlowConfig(
            c.dbl("tcp_idle_timeout_s"), c.dbl("udp_idle_timeout_s"), c.dbl("other_idle_timeout_s"), c.dbl("max_flow_duration_s"),
            c.dbl("tcp_time_wait_s"), c.int("max_packets_per_flow"), c.int("drop_tcp_retransmissions") != 0, c.dbl("expire_check_interval_s"),
        )
    }
}

/** Port of src/data/flow_tracker.py (same splitting, initiator, retransmission and cap rules). */
class FlowTracker(private val cfg: FlowConfig) {
    private val active = LinkedHashMap<FlowKey, Flow>()
    private var finished = ArrayList<Flow>()
    private var nextIndex = 0
    private var lastSweepTs: Double? = null

    val activeCount get() = active.size

    private fun makeKey(p: ParsedPacket): FlowKey {
        val a = Endpoint(p.srcIp, p.sport)
        val b = Endpoint(p.dstIp, p.dport)
        return if (a <= b) FlowKey(p.ipProto, a, b) else FlowKey(p.ipProto, b, a)
    }

    private fun seqLe(a: Long, b: Long) = ((b - a) and 0xFFFFFFFFL) < 0x80000000L

    /** Returns the Flow the packet was appended to, or null if it was not kept (dropped retransmission / cap). */
    fun process(pkt: ParsedPacket): Flow? {
        maybeSweep(pkt.ts)
        val key = makeKey(pkt)
        var flow = active[key]
        if (flow != null) {
            val reason = shouldSplit(flow, pkt)
            if (reason != null) {
                finish(flow)
                active.remove(key)
                flow = startFlow(key, pkt, reason == "max_duration", flow)
            }
        } else {
            flow = startFlow(key, pkt, false, null)
        }
        val before = flow.packets.size
        addPacket(flow, pkt)
        return if (flow.packets.size > before) flow else null
    }

    fun drainFinished(): List<Flow> { val d = finished; finished = ArrayList(); return d }

    fun evictOldest(keep: Int): Int {
        val extra = active.size - keep
        if (extra <= 0) return 0
        for (k in active.keys.sortedBy { active[it]!!.lastTs }.take(extra)) finish(active.remove(k)!!)
        return extra
    }

    fun flush(): List<Flow> {
        for (f in active.values.toList()) finish(f)
        active.clear()
        val out = finished.sortedBy { it.flowIndex }
        finished = ArrayList()
        return out
    }

    private fun idleTimeout(cat: Int) = when (cat) {
        PROTO_TCP -> cfg.tcpIdleTimeoutS
        PROTO_UDP, PROTO_DNS -> cfg.udpIdleTimeoutS
        else -> cfg.otherIdleTimeoutS
    }

    private fun shouldSplit(flow: Flow, pkt: ParsedPacket): String? {
        if (pkt.ts - flow.lastTs > idleTimeout(flow.category)) return "idle_timeout"
        if (pkt.ts - flow.startTs > cfg.maxFlowDurationS) return "max_duration"
        if (flow.category == PROTO_TCP || pkt.ipProto == 6) {
            val pureSyn = (pkt.tcpFlags and TCP_SYN) != 0 && (pkt.tcpFlags and TCP_ACK) == 0
            if (pureSyn) {
                val sameSide = Endpoint(pkt.srcIp, pkt.sport) == flow.initiator
                val isRetxOfFirst = sameSide && flow.synSeq != null && pkt.seq == flow.synSeq
                if (!isRetxOfFirst) {
                    if (flow.closed || flow.synSeen || flow.packets.size > 0) return "new_syn"
                }
            }
            if (flow.closed && (pkt.ts - flow.closedTs) > cfg.tcpTimeWaitS) return "after_close"
        }
        return null
    }

    private fun startFlow(key: FlowKey, pkt: ParsedPacket, continued: Boolean, prev: Flow?): Flow {
        val src = Endpoint(pkt.srcIp, pkt.sport)
        val dst = Endpoint(pkt.dstIp, pkt.dport)
        var initiator = src
        var responder = dst
        if (continued && prev != null) {
            initiator = prev.initiator; responder = prev.responder
        } else if (pkt.ipProto == 6) {
            val syn = (pkt.tcpFlags and TCP_SYN) != 0
            val ack = (pkt.tcpFlags and TCP_ACK) != 0
            if (syn && ack) { initiator = dst; responder = src }
        }
        val flow = Flow(nextIndex, key, initiator, responder, pkt.ipProto, pkt.category, pkt.ts, pkt.ts)
        nextIndex++
        active[key] = flow
        return flow
    }

    private fun addPacket(flow: Flow, pkt: ParsedPacket) {
        flow.lastTs = maxOf(flow.lastTs, pkt.ts)
        val direction = if (Endpoint(pkt.srcIp, pkt.sport) == flow.initiator) 1 else -1
        if (pkt.ipProto == 6) {
            val flags = pkt.tcpFlags
            if ((flags and TCP_SYN) != 0 && (flags and TCP_ACK) == 0 && !flow.synSeen) { flow.synSeen = true; flow.synSeq = pkt.seq }
            if ((flags and TCP_FIN) != 0) flow.finDir.add(direction)
            if ((flags and TCP_RST) != 0) close(flow, pkt.ts)
            else if (flow.finDir.size == 2) close(flow, pkt.ts)
            if (pkt.payloadLen > 0 && (flags and (TCP_SYN or TCP_RST)) == 0) {
                val end = (pkt.seq + pkt.payloadLen) and 0xFFFFFFFFL
                val prevEnd = flow.maxEndSeq[direction]
                if (prevEnd != null && seqLe(end, prevEnd)) {
                    flow.retransmissions++
                    if (cfg.dropTcpRetransmissions) return
                }
                if (prevEnd == null || !seqLe(end, prevEnd)) flow.maxEndSeq[direction] = end
            } else if ((flags and TCP_SYN) != 0) {
                flow.maxEndSeq[direction] = (pkt.seq + 1) and 0xFFFFFFFFL
            }
        }
        if (flow.packets.size >= cfg.maxPacketsPerFlow) { flow.packetsDroppedCap++; return }
        flow.packets.add(PacketRecord(pkt.ts, direction, pkt.category, pkt.payloadLen, pkt.payload, pkt.tcpFlags))
    }

    private fun close(flow: Flow, ts: Double) { if (!flow.closed) { flow.closed = true; flow.closedTs = ts } }

    private fun finish(flow: Flow) { finished.add(flow) }

    private fun maybeSweep(now: Double) {
        val last = lastSweepTs
        if (last == null) { lastSweepTs = now; return }
        if (now - last < cfg.expireCheckIntervalS) return
        lastSweepTs = now
        val expired = active.entries.filter { now - it.value.lastTs > idleTimeout(it.value.category) }.map { it.key }
        for (k in expired) finish(active.remove(k)!!)
    }
}
