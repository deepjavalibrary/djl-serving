plugins {
    ai.djl.javaProject
}

val exclusion by configurations.registering

@Suppress("UnstableApiUsage")
dependencies {
    api(platform("ai.djl:bom:${version}"))
    api(platform("software.amazon.awssdk:bom:${libs.versions.awssdk.get()}"))
    api("ai.djl.aws:aws-ai")
    api("software.amazon.awssdk:dynamodb")
    api("software.amazon.awssdk:s3")

    implementation(project(":serving"))

    testImplementation("com.amazonaws:DynamoDBLocal:2.4.0")
    testImplementation("cloud.localstack:localstack-utils:0.2.15")
    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }

    exclusion(project(":serving"))
    exclusion("com.google.code.gson:gson")
}

tasks {
    jar {
        finalizedBy("copyJar")
        includeEmptyDirs = false
        duplicatesStrategy = DuplicatesStrategy.INCLUDE
        from((configurations.runtimeClasspath.get() - exclusion.get()).map {
            if (it.isDirectory()) it else zipTree(it)
        })
    }

    register<Copy>("copyNativeDeps") {
        from(configurations.testRuntimeClasspath) {
            include("*.dylib")
            include("*.so")
            include("*.dll")
        }
        into("build/native")
    }

    register<Copy>("copyJar") {
        from(jar) // here it automatically reads jar file produced from jar task
        into("../../serving/plugins")
    }

    test {
        dependsOn("copyNativeDeps")
        systemProperty("java.library.path", "build/native")
    }
}
