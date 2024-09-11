plugins {
    ai.djl.javaProject
    id("com.google.protobuf") version "0.9.4"
}

val exclusion by configurations.registering

@Suppress("UnstableApiUsage")
dependencies {
    api(platform("ai.djl:bom:${version}"))
    implementation(project(":serving"))
    implementation("io.grpc:grpc-netty-shaded:${libs.versions.grpc.get()}")
    implementation("io.grpc:grpc-protobuf:${libs.versions.grpc.get()}")
    implementation("io.grpc:grpc-stub:${libs.versions.grpc.get()}")
    implementation("io.grpc:protoc-gen-grpc-java:${libs.versions.grpc.get()}")
    // necessary for Java 9+
    compileOnly("org.apache.tomcat:annotations-api:${libs.versions.annotationsApi.get()}")

    testImplementation(libs.commons.cli)
    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }

    exclusion(project(":serving"))
    exclusion("com.google.code.gson:gson")
}

protobuf {
    protoc {
        artifact = "com.google.protobuf:protoc:${libs.versions.protoc.get()}"
    }
    plugins {
        create("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:${libs.versions.grpc.get()}"
        }
    }
    generateProtoTasks {
        all().forEach {
            it.plugins {
                create("grpc")
            }
        }
    }
}

tasks {
    processResources {
        dependsOn(generateProto)
    }

    jar {
        includeEmptyDirs = false
        duplicatesStrategy = DuplicatesStrategy.INCLUDE
        from((configurations.runtimeClasspath.get() - exclusion.get()).map {
            if (it.isDirectory()) it else zipTree(it)
        })
    }

    verifyJava {
        dependsOn(generateProto)
    }
    checkstyleMain { exclude("ai/djl/serving/grpc/proto/*") }
    pmdMain { exclude("ai/djl/serving/grpc/proto/*") }
}
