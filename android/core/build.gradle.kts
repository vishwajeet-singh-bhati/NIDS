plugins {
    id("org.jetbrains.kotlin.jvm")
    application
}
application { mainClass.set("nids.core.GoldenCheck") }   // gradle :core:run --args="../golden"
