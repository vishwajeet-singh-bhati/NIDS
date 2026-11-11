plugins {
    id("org.jetbrains.kotlin.jvm")
    application
}
kotlin { jvmToolchain(17) }
application { mainClass.set("nids.core.GoldenCheck") }   // gradle :core:run --args="../golden"
