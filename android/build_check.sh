#!/usr/bin/env bash
# Compile the pure-Kotlin core with the Kotlin compiler that ships inside Gradle and run the parity check.
#   android/build_check.sh <golden dir> [gradle lib dir]
set -euo pipefail
GOLD=${1:?golden dir}; LIB=${2:-/opt/gradle-8.14.3/lib}
HERE=$(cd "$(dirname "$0")" && pwd); OUT=$(mktemp -d)
CP=$(ls $LIB/kotlin-compiler-embeddable-*.jar $LIB/kotlin-stdlib-2*.jar $LIB/kotlin-script-runtime-*.jar $LIB/kotlin-reflect-*.jar \
   $LIB/kotlin-daemon-embeddable-*.jar $LIB/trove4j-*.jar $LIB/annotations-24*.jar $LIB/kotlinx-coroutines-core-jvm-*.jar | tr '\n' ':')
java -cp "$CP" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib -no-reflect -cp "$(ls $LIB/kotlin-stdlib-2*.jar)" \
   -d "$OUT" $HERE/core/src/main/kotlin/nids/core/*.kt 2>&1 | grep -v "^Picked up" || true
java -cp "$OUT:$(ls $LIB/kotlin-stdlib-2*.jar)" nids.core.GoldenCheck "$GOLD"
