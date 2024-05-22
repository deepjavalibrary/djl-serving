import com.github.gradle.node.npm.task.NpmTask

plugins {
    ai.djl.javaProject
    id("com.github.node-gradle.node") version "3.4.0"
}

dependencies {
    api(project(":serving"))
}

tasks {
    register<NpmTask>("buildConsoleApp") {
        dependsOn("npmInstall")
        project.logger.info("Build the DJL Management console application")
        npmCommand = listOf("run", "build")
    }

    register<Copy>("copyJar") {
        from(jar) // here it automatically reads jar file produced from jar task
        into("../../serving/plugins")
    }

    jar { finalizedBy("copyJar") }

    node {
        download = true
        nodeProjectDir = projectDir / "webapp"
    }

    clean {
        doFirst {
            delete("webapp/dist")
            delete("src/main/resources/static/console")
        }
    }

    if (projectDir.toString() == System.getProperty("user.dir") ||
        !(projectDir / "src/main/resources/static/console").exists() &&
        "win" !in os) {
        processResources {
            // Run npm task only when running gradle in current directory on Windows
            dependsOn("buildConsoleApp")
        }
    }
}
