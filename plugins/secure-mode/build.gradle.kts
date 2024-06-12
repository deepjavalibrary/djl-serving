plugins {
    ai.djl.javaProject
}

dependencies {
    implementation(project(":serving"))
    implementation(project(":wlm"))

    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
    testImplementation("org.mockito:mockito-core")

}

tasks {
    register<Copy>("copyJar") {
        from(jar) // here it automatically reads jar file produced from jar task
        into("../../serving/plugins")
    }

    jar { finalizedBy("copyJar") }
}
