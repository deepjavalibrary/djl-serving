plugins {
    ai.djl.javaProject
}

dependencies {
    implementation(platform("ai.djl:bom:${version}"))
    implementation(libs.huggingface.tokenizers) {
        exclude(group = "org.apache.commons", module = "commons-compress")
    }
    implementation(libs.slf4j.simple)
    implementation(libs.commons.cli)
    implementation(libs.commons.codec)
    implementation(libs.apache.httpclient)
    implementation(libs.apache.httpmime)
    implementation("com.jayway.jsonpath:json-path:${libs.versions.jsonpath.get()}") {
        exclude(group = "net.minidev", module = "json-smart")
    }

    testImplementation(libs.netty.http)
    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }

    tasks {
        jar {
            manifest {
                attributes["Main-Class"] = "ai.djl.awscurl.AwsCurl"
            }
            includeEmptyDirs = false
            duplicatesStrategy = DuplicatesStrategy.EXCLUDE
            from(configurations.runtimeClasspath.get().map {
                if (it.isDirectory()) it else zipTree(it).matching {
                    exclude("**/*.so")
                    exclude("**/*.dylib")
                    exclude("**/*.dll")
                }
            })

            doLast {
                providers.exec {
                    workingDir = projectDir
                    commandLine("sh")
                    args(
                        "-c",
                        "cat src/main/scripts/stub.sh build/libs/awscurl*.jar > build/awscurl && chmod +x build/awscurl"
                    )
                }.result.get()
            }
        }
    }
}
