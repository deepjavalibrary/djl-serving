plugins {
    id("base")
    id("jacoco-report-aggregation")
}

repositories {
    mavenCentral()
    maven("https://oss.sonatype.org/content/repositories/snapshots/")
}

dependencies {
    jacocoAggregation(project(":awscurl"))
    jacocoAggregation(project(":benchmark"))
    jacocoAggregation(project(":engines:python"))
    jacocoAggregation(project(":plugins:cache"))
    jacocoAggregation(project(":plugins:kserve"))
    // jacocoAggregation(project(":plugins:management-console"))
    jacocoAggregation(project(":plugins:plugin-management-plugin"))
    jacocoAggregation(project(":plugins:secure-mode"))
    jacocoAggregation(project(":plugins:static-file-plugin"))
    jacocoAggregation(project(":prometheus"))
    jacocoAggregation(project(":serving"))
    jacocoAggregation(project(":wlm"))
}

tasks {
    reporting {
        @Suppress("UnstableApiUsage")
        reports {
            val testCodeCoverageReport by creating(JacocoCoverageReport::class) {
                testType = TestSuiteType.UNIT_TEST
            }
        }
    }

    check {
        dependsOn("testCodeCoverageReport")
    }
}
