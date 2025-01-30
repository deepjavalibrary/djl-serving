package ai.djl

open class Cmd @Inject constructor(@Internal val execOperations: ExecOperations) : DefaultTask()

tasks {
    register<Cmd>("formatShell") {
        doLast {
            execOperations.exec {
                workingDir = projectDir
                commandLine(
                    "bash",
                    "-c",
                    "find tests serving -name '*.sh' | xargs shfmt -i 2 -w"
                )
            }
        }
    }
}