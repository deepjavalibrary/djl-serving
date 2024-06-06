plugins {
    ai.djl.javaProject
}

dependencies {
    implementation(platform("ai.djl:bom:${version}"))
    api("ai.djl:api")

    implementation(libs.prometheus.core)
    implementation(libs.prometheus.exposition.formats) {
        exclude(group = "io.prometheus", module = "prometheus-metrics-shaded-protobuf")
    }
    implementation(libs.apache.log4j.core)
    annotationProcessor(libs.apache.log4j.core)

    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
    testRuntimeOnly(libs.apache.log4j.slf4j)
}

tasks {
    test {
        systemProperty("SERVING_PROMETHEUS", "true")
        systemProperty("log4j.configurationFile", "${projectDir}/src/test/resources/log4j2.xml")
    }
}
