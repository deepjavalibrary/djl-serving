package ai.djl

open class Cmd @Inject constructor(@Internal val execOperations: ExecOperations) : DefaultTask()

tasks {
    register<Cmd>("formatPython") {
        doLast {
            execOperations.exec {
                workingDir = projectDir
                commandLine(
                    "bash",
                    "-c",
                    "find . -name '*.py' -not -path '*/.gradle/*' -not -path '*/build/*' -not -path '*/venv/*' -print0 | xargs -0 yapf --in-place"
                )
            }
        }
    }

    register<Cmd>("verifyPython") {
        doFirst {
            try {
                execOperations.exec {
                    workingDir = projectDir
                    commandLine(
                        "bash",
                        "-c",
                        "find . -name '*.py' -not -path '*/.gradle/*' -not -path '*/build/*' -not -path '*/venv/*' -not -path '*/tests/integration/examples/custom_formatters/load_formatter_failed.py' -print0 | xargs -0 yapf -d"
                    )
                }
            } catch (e: Exception) {
                throw GradleException(
                    "Repo is improperly formatted, please run ./gradlew formatPython, and recommit",
                    e
                )
            }
        }
    }
}