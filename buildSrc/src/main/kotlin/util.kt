import org.gradle.api.Project
import org.gradle.api.file.Directory
import java.io.File
import java.net.URI
import java.net.URL

operator fun File.div(other: String) = File(this, other)
operator fun Directory.div(other: String): File = file(other).asFile

infix fun URL.into(file: File) {
    file.outputStream().use { out ->
        openStream().use { `in` -> `in`.copyTo(out) }
    }
}

var File.text
    get() = readText()
    set(value) = writeText(value)

val URL.text
    get() = readText()

val osName: String = System.getProperty("os.name")
val os = osName.lowercase()
val home: String = System.getProperty("user.home")
val hasGpu = runCatching { Runtime.getRuntime().exec(arrayOf("nvidia-smi", "-L")).waitFor() == 0 }.getOrElse { false }

val String.url: URL
    get() = URI(this).toURL()

// provide directly a Directory instead of a DirectoryProperty
val Project.buildDirectory: Directory
    get() = layout.buildDirectory.get()