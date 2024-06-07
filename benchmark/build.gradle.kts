@file:Suppress("UNCHECKED_CAST")

import com.netflix.gradle.plugins.deb.Deb

plugins {
    ai.djl.javaProject
    application
    id("com.netflix.nebula.ospackage") version "11.4.0"
}

dependencies {
    implementation(platform("ai.djl:bom:${version}"))
    implementation(project(":wlm"))

    implementation(libs.commons.cli)
    implementation(libs.apache.log4j.slf4j)
    implementation("ai.djl:model-zoo")

    runtimeOnly("ai.djl.pytorch:pytorch-model-zoo")
    runtimeOnly("ai.djl.tensorflow:tensorflow-model-zoo")
    runtimeOnly("ai.djl.mxnet:mxnet-model-zoo")
    runtimeOnly("ai.djl.ml.xgboost:xgboost")
    runtimeOnly("ai.djl.tensorrt:tensorrt")
    runtimeOnly("ai.djl.huggingface:tokenizers")
    runtimeOnly(project(":engines:python"))

    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }

    if (hasGpu) {
        runtimeOnly("ai.djl.onnxruntime:onnxruntime-engine") {
            exclude(group = "com.microsoft.onnxruntime", module = "onnxruntime")
        }
        runtimeOnly(libs.onnxruntime.gpu)
    } else {
        runtimeOnly("ai.djl.onnxruntime:onnxruntime-engine")
    }
}

tasks {
    application {
        mainClass = System.getProperty("main", "ai.djl.benchmark.Benchmark")
    }

    run.configure {
        environment("TF_CPP_MIN_LOG_LEVEL", "1") // turn off TensorFlow print out
        systemProperties = System.getProperties().toMap() as Map<String, Any>
        systemProperties.remove("user.dir")
    }

    register<JavaExec>("benchmark") {
        environment("TF_CPP_MIN_LOG_LEVEL", "1") // turn off TensorFlow print out
        val arguments = gradle.startParameter.taskRequests[0].args
        for (argument in arguments) {
            if (argument.trim().startsWith("--args")) {
                var line = argument.split("=", limit = 2)
                if (line.size == 2) {
                    line = line[1].split(" ")
                    if (line.contains("-t")) {
                        if (System.getProperty("ai.djl.default_engine") == "TensorFlow") {
                            environment("OMP_NUM_THREADS", "1")
                            environment("TF_NUM_INTRAOP_THREADS", "1")
                        } else {
                            environment("MXNET_ENGINE_TYPE", "NaiveEngine")
                            environment("OMP_NUM_THREADS", "1")
                        }
                    }
                    break
                }
            }
        }

        systemProperties = System.getProperties().toMap() as Map<String, Any>
        systemProperties.remove("user.dir")
        classpath = sourceSets.main.get().runtimeClasspath
        // restrict the jvm heap size for better monitoring benchmark
        if (project.hasProperty("loggc")) {
            jvmArgs("-Xmx2g", "-Xlog:gc*=debug:file=build/gc.log")
        } else {
            jvmArgs("-Xmx2g")
        }
        mainClass = "ai.djl.benchmark.Benchmark"
    }

    register("prepareDeb") {
        dependsOn(distTar)
        doFirst {
            exec {
                commandLine(
                    "tar",
                    "xvf",
                    "${buildDirectory}/distributions/benchmark-${version}.tar",
                    "-C",
                    "$buildDirectory"
                )
            }
        }
    }

    register<Deb>("createDeb") {
        dependsOn("prepareDeb")

        packageName = "djl-bench"
        archiveVersion = version
        release = "1"
        maintainer = "Deep Java Library <djl-dev@amazon.com>"
        summary = "djl-bench is a command line tool that allows you to benchmark the\n" +
                "  model on all different platforms for single-thread/multi-thread\n" +
                "  inference performance."

        from(buildDirectory / "benchmark-${version}") {
            into("/usr/local/djl-bench-${version}")
        }
        link("/usr/bin/djl-bench", "/usr/local/djl-bench-${version}/bin/benchmark")
    }

    register<Exec>("installOnLinux") {
        dependsOn("createDeb")
        doFirst {
            if ("linux" in os) {
                val ver = version.toString().replace("-", "~")
                commandLine("sudo", "dpkg", "-i", "${buildDirectory}/distributions/djl-bench_${ver}-1_all.deb")
            } else {
                throw GradleException("task installOnLinux Only supported on Linux.")
            }
        }
    }

    startScripts {
        doLast {
            val replacement = "CLASSPATH=\\\$APP_HOME/lib/*\n\n" +
                    "if [[ \"\\\$*\" == *-t* || \"\\\$*\" == *--threads* ]]\n" +
                    "then\n" +
                    "    export TF_CPP_MIN_LOG_LEVEL=1\n" +
                    "    export MXNET_ENGINE_TYPE=NaiveEngine\n" +
                    "    export OMP_NUM_THREADS=1\n" +
                    "    export TF_NUM_INTRAOP_THREADS=1\n" +
                    "fi"

            unixScript.text = unixScript.text
                .replace(Regex("CLASSPATH=\\\$APP_HOME/lib/.*"), replacement)
                .replace("/usr/bin/env sh", "/usr/bin/env bash")
                .replace("#!/bin/sh", "#!/bin/bash")
        }
    }
}
