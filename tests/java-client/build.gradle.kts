plugins {
    `kotlin-dsl`
    java
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    mavenLocal()
    maven("https://oss.sonatype.org/content/repositories/snapshots/")
}

dependencies {
    implementation(platform("ai.djl:bom:${libs.versions.djl.get()}"))
    implementation("ai.djl:api")
    implementation(libs.slf4j.api)

    testImplementation(libs.slf4j.simple)
    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
}

tasks {
    test {
        useTestNG {
            //suiteXmlFiles = listOf(File(rootDir, "testng.xml")) //This is how to add custom testng.xml
        }
    }
}