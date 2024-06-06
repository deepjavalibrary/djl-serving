plugins {
    ai.djl.javaProject
}

dependencies {
    api(platform("ai.djl:bom:${version}"))
    api("ai.djl:api")
    api(libs.slf4j.api)
    api(libs.snakeyaml)

    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }

    testRuntimeOnly("ai.djl:model-zoo")
    testRuntimeOnly("ai.djl.pytorch:pytorch-engine")
    testRuntimeOnly(libs.slf4j.simple)
}
