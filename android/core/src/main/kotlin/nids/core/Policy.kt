package nids.core

class PolicyParams(val threshold: Double, val emaAlpha: Double, val minHits: Int, val hardMargin: Double)

class PolicyState {
    var ema: Double? = null
    var hits = 0
    var scored = 0
    var alerted = false
}

/** Flow alert policy (src/stream/policy.py): EMA of window logits + min consecutive hits + hard margin. */
object Policy {
    fun step(st: PolicyState, z: Double, p: PolicyParams): String? {
        st.scored++
        val e = st.ema
        st.ema = if (e == null) z else p.emaAlpha * z + (1 - p.emaAlpha) * e
        st.hits = if (z >= p.threshold) st.hits + 1 else 0
        if (st.alerted) return null
        var reason: String? = null
        if (z >= p.threshold + p.hardMargin) reason = "single_window_high"
        else if (st.hits >= p.minHits && st.ema!! >= p.threshold) reason = "sustained"
        if (reason != null) st.alerted = true
        return reason
    }
}

/** Device alarm (src/stream/device_rule.py): >= minAlerts flow alerts within windowS seconds, then quiet for windowS. */
class BurstDetector(private val windowS: Double, private val minAlerts: Int) {
    private val times = ArrayDeque<Double>()
    private var quietUntil = Double.NEGATIVE_INFINITY
    fun add(t: Double): Boolean {
        times.addLast(t)
        while (times.isNotEmpty() && t - times.first() > windowS) times.removeFirst()
        if (times.size >= minAlerts && t >= quietUntil) { quietUntil = t + windowS; return true }
        return false
    }
}
