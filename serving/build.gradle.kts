@file:Suppress("UNCHECKED_CAST")

import com.netflix.gradle.plugins.deb.Deb

plugins {
    ai.djl.javaProject
    ai.djl.publish
    application
    id("com.netflix.nebula.ospackage") version "11.4.0"
}

dependencies {
    api(platform("ai.djl:bom:${version}"))
    api(project(":wlm"))
    api("io.netty:netty-codec-http:${libs.versions.netty.get()}")
    api("io.netty:netty-transport-native-epoll:${libs.versions.netty.get()}:linux-aarch_64")
    api("io.netty:netty-transport-native-epoll:${libs.versions.netty.get()}:linux-x86_64")
    api("io.netty:netty-transport-native-kqueue:${libs.versions.netty.get()}:osx-aarch_64")
    api("io.netty:netty-transport-native-kqueue:${libs.versions.netty.get()}:osx-x86_64")

    //noinspection GradlePackageUpdate
    implementation(libs.commons.cli)
    implementation(project(":prometheus"))

    runtimeOnly(libs.apache.log4j.slf4j)
    runtimeOnly(libs.disruptor)

    runtimeOnly("ai.djl:model-zoo")
    runtimeOnly("ai.djl.tensorflow:tensorflow-model-zoo")
    runtimeOnly("ai.djl.pytorch:pytorch-model-zoo")
    runtimeOnly("ai.djl.huggingface:tokenizers")
    runtimeOnly("ai.djl.tensorrt:tensorrt")
    runtimeOnly(project(":engines:python"))

    if (hasGpu) {
        runtimeOnly("ai.djl.onnxruntime:onnxruntime-engine") {
            exclude(group = "com.microsoft.onnxruntime", module = "onnxruntime")
        }
        runtimeOnly(libs.onnxruntime.gpu)
    } else {
        runtimeOnly("ai.djl.onnxruntime:onnxruntime-engine")
    }

    testRuntimeOnly("org.bouncycastle:bcpkix-jdk18on:1.78")
    testRuntimeOnly("org.bouncycastle:bcprov-jdk18on:1.78")
    testRuntimeOnly(libs.snakeyaml)
    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
}

tasks {
    jar {
        manifest {
            attributes["Main-Class"] = "ai.djl.serving.ModelServer"
        }
        includeEmptyDirs = false

        exclude("META-INF/maven/**")
        exclude("META-INF/INDEX.LIST")
        exclude("META-INF/MANIFEST*")
    }

    test {
        dependsOn(
            ":plugins:kserve:jar",
            ":plugins:management-console:jar",
            ":plugins:plugin-management-plugin:jar",
            ":plugins:static-file-plugin:jar"
        )
        workingDir(projectDir)
        systemProperty("SERVING_PROMETHEUS", "true")
        systemProperty("log4j.configurationFile", "${projectDir}/src/main/conf/log4j2.xml")
    }

    application {
        mainClass = System.getProperty("main", "ai.djl.serving.ModelServer")

        applicationDistribution.into("conf") {
            from("src/main/conf/")
        }
        applicationDistribution.into("plugins") {
            from(projectDir / "plugins")
        }
    }

    distTar {
        dependsOn(
            ":plugins:cache:copyJar",
            ":plugins:kserve:copyJar",
            ":plugins:management-console:copyJar",
            ":plugins:plugin-management-plugin:copyJar",
            ":plugins:static-file-plugin:copyJar",
            ":plugins:secure-mode-plugin:copyJar"
        )
    }

    distZip {
        dependsOn(
            ":plugins:cache:copyJar",
            ":plugins:kserve:copyJar",
            ":plugins:management-console:copyJar",
            ":plugins:plugin-management-plugin:copyJar",
            ":plugins:static-file-plugin:copyJar",
            ":plugins:secure-mode-plugin:copyJar"
        )
    }

    run.configure {
        dependsOn(
            ":plugins:kserve:jar",
            ":plugins:management-console:jar",
            ":plugins:static-file-plugin:jar"
        )
        environment("TF_CPP_MIN_LOG_LEVEL", "1") // turn off TensorFlow print out
        environment("MXNET_ENGINE_TYPE", "NaiveEngine")
        environment("OMP_NUM_THREADS", "1")
        environment("MODEL_SERVER_HOME", "$projectDir")
        systemProperties = System.getProperties().toMap() as Map<String, Any>
        systemProperties.remove("user.dir")
        systemProperty("SERVING_PROMETHEUS", "true")
        systemProperty("log4j.configurationFile", "${projectDir}/src/main/conf/log4j2.xml")
        application.applicationDefaultJvmArgs =
            listOf("-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=4000")
        workingDir(projectDir)
    }

    clean {
        doFirst {
            delete("plugins")
            delete("docker/distributions")
            delete("logs")
        }
    }

    startScripts {
        doLast {
            unixScript.text = unixScript.text.replace(
                "exec \"\$JAVACMD\" \"\$@\"",
                "if [ -f \"/opt/djl/bin/telemetry.sh\" ]; then\n" +
                        "    /opt/djl/bin/telemetry.sh\n" +
                        "fi\n" +
                        "if [ \"\${OMP_NUM_THREADS}\" = \"\" ] && [ \"\${NO_OMP_NUM_THREADS}\" = \"\" ] ; then\n" +
                        "    export OMP_NUM_THREADS=1\n" +
                        "fi\n" +
                        "if [ \"\${TF_CPP_MIN_LOG_LEVEL}\" = \"\" ] ; then\n" +
                        "    export TF_CPP_MIN_LOG_LEVEL=1\n" +
                        "fi\n" +
                        "if [ \"\${TF_NUM_INTRAOP_THREADS}\" = \"\" ] ; then\n" +
                        "    export TF_NUM_INTRAOP_THREADS=1\n" +
                        "fi\n" +
                        "exec env MXNET_ENGINE_TYPE=\"NaiveEngine\" \"\$JAVACMD\" \"\$@\""
            ).replace(
                "DEFAULT_JVM_OPTS=\"\"",
                "if [ \"\${MODEL_SERVER_HOME}\" = \"\" ] ; then\n" +
                        "    export MODEL_SERVER_HOME=\${APP_HOME}\n" +
                        "fi\n" +
                        "if [ -f \"/opt/ml/.sagemaker_infra/endpoint-metadata.json\" ]; then\n" +
                        "    export JAVA_OPTS=\"\$JAVA_OPTS -XX:-UseContainerSupport\"\n" +
                        "    DEFAULT_JVM_OPTS=\"\${DEFAULT_JVM_OPTS:--Dlog4j.configurationFile=\${APP_HOME}/conf/log4j2-plain.xml}\"\n" +
                        "else\n" +
                        "    DEFAULT_JVM_OPTS=\"\${DEFAULT_JVM_OPTS:--Dlog4j.configurationFile=\${APP_HOME}/conf/log4j2.xml}\"\n" +
                        "fi\n"
            ).replace(Regex("CLASSPATH=\\\$APP_HOME/lib/.*"), "CLASSPATH=\\\$APP_HOME/lib/*")
        }
    }

    register("prepareDeb") {
        dependsOn(distTar)
        doFirst {
            exec {
                commandLine(
                    "tar",
                    "xvf",
                    "${buildDirectory}/distributions/serving-${version}.tar",
                    "-C",
                    "$buildDirectory"
                )
            }
        }
    }

    register<Deb>("createDeb") {
        dependsOn("prepareDeb")
        packageName = "djl-serving"
        archiveVersion = "$project.version"
        release = "1"
        maintainer = "Deep Java Library <djl-dev@amazon.com>"
        summary = "djl-serving is a general model server that can serve both Deep Learning models" +
                "and traditional machine learning models."

        postInstall(
            "mkdir -p /usr/local/djl-serving-${project.version}/models" +
                    " && mkdir -p /usr/local/djl-serving-${project.version}/plugins"
        )

        from(buildDirectory / "serving-${project.version}") {
            into("/usr/local/djl-serving-${project.version}")
        }
        link("/usr/bin/djl-serving", "/usr/local/djl-serving-${project.version}/bin/serving")
    }

    register<Copy>("dockerDeb") {
        dependsOn("createDeb")
        from(buildDirectory / "distributions")
        include("*.deb")
        into(projectDir / "docker/distributions")
    }
}
