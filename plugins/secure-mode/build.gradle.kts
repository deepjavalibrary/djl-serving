plugins {
    ai.djl.javaProject
}

dependencies {
    // Note: We only depend on wlm, not serving, to avoid circular dependencies
    // serving -> secure-mode -> serving (circular!)
    implementation(project(":wlm"))
    
    // For IllegalConfigurationException
    compileOnly(project(":serving"))

    testImplementation(project(":serving"))
    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
    testImplementation(libs.mockito.core)

}

tasks {
    register<Copy>("copyJar") {
        from(jar) // here it automatically reads jar file produced from jar task
        into("../../serving/plugins")
    }

    jar { finalizedBy("copyJar") }
}
