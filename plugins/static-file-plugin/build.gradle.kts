plugins {
    ai.djl.javaProject
}

dependencies {
    implementation(project(":serving"))

    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
}

tasks {
    register<Copy>("copyJar") {
        from(jar) // here it automatically reads jar file produced from jar task
        into("../../serving/plugins")
    }

    jar { finalizedBy("copyJar") }
}
