package ai.djl

import org.gradle.accessors.dm.LibrariesForLibs
import org.gradle.kotlin.dsl.java
import org.gradle.kotlin.dsl.maven
import org.gradle.kotlin.dsl.repositories
import org.gradle.kotlin.dsl.the

plugins {
    java
}

val libs = the<LibrariesForLibs>()
var servingVersion: String? = System.getenv("DJL_VERSION")
val stagingRepo: String? = System.getenv("DJL_STAGING")
servingVersion = if (servingVersion == null) libs.versions.djl.get() else servingVersion
if (!project.hasProperty("staging")) {
    servingVersion += "-SNAPSHOT"
}

group = "ai.djl.serving"
version = servingVersion!!

repositories {
    mavenCentral()
    mavenLocal()
    maven("https://oss.sonatype.org/service/local/repositories/${stagingRepo}/content/")
    maven("https://oss.sonatype.org/content/repositories/snapshots/")
}
