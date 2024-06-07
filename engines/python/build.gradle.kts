plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.python"

fun stripPackageVersion() {
    val initFile = projectDir / "setup/djl_python/__init__.py"
    initFile.text = initFile.text.replace(Regex("\\n*__version__.*"), "")
}

dependencies {
    api(platform("ai.djl:bom:${version}"))
    api("ai.djl:api")
    api("io.netty:netty-codec:${libs.versions.netty.get()}")
    api("io.netty:netty-handler:${libs.versions.netty.get()}")
    api("io.netty:netty-transport:${libs.versions.netty.get()}")
    api("io.netty:netty-transport-native-epoll:${libs.versions.netty.get()}:linux-aarch_64")
    api("io.netty:netty-transport-native-epoll:${libs.versions.netty.get()}:linux-x86_64")
    api("io.netty:netty-transport-native-kqueue:${libs.versions.netty.get()}:osx-aarch_64")
    api("io.netty:netty-transport-native-kqueue:${libs.versions.netty.get()}:osx-x86_64")
    api(libs.slf4j.api)

    testImplementation(libs.slf4j.simple)
    testImplementation(libs.testng)
}

tasks {
    sourceSets {
        main {
            resources {
                srcDirs("setup")
            }
        }
    }

    processResources {
        val path = projectDir / "build/resources/main"
        inputs.properties(mapOf("djl_version" to libs.versions.djl.get()))
        outputs.dir("$path/native/lib")
        doFirst {
            stripPackageVersion()
            val f = file("setup/djl_python/__init__.py")
            f.text += "\n__version__ = '${libs.versions.djl.get()}'\n"
        }

        exclude("build", "*.egg-info", "__pycache__", "PyPiDescription.rst", "setup.py", "djl_python/tests", "*.pt")
        doLast {
            val list = ArrayList<String>()
            val dir = projectDir / "setup/djl_python"
            dir.walkTopDown().forEach {
                val name = it.relativeTo(dir).toString()
                if (it.isFile() && !name.contains("__pycache__") && !name.contains("tests")) {
                    list.add(name)
                }
            }
            list.sort()

            val sb = StringBuilder()
            sb.append("version=${version}\nlibraries=djl_python_engine.py")
            for (name in list) {
                sb.append(",djl_python/").append(name.replace("\\", "/"))
            }
            // write properties
            val propFile = path / "native/lib/python.properties"
            propFile.parentFile.mkdirs()
            propFile.text = sb.toString()
        }
    }

    test {
        environment("TENSOR_PARALLEL_DEGREE", "2")
        systemProperty("org.slf4j.simpleLogger.log.io.netty", "warn")
    }

    clean {
        doFirst {
            delete("setup/build/")
            delete("setup/djl_python.egg-info/")
            delete("setup/__pycache__/")
            delete("setup/djl_python/__pycache__/")
            delete("setup/djl_python/tests/__pycache__/")
            delete("setup/djl_python/scheduler/__pycache__/")
            delete("src/test/resources/accumulate/__pycache__/")
            delete("$home/.djl.ai/python")
            stripPackageVersion()
        }
    }

    register("cleanVersion") {
        doFirst {
            stripPackageVersion()
        }
    }

    jar { finalizedBy("cleanVersion") }
}
