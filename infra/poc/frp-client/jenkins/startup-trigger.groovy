import jenkins.model.Jenkins
import hudson.model.FreeStyleProject
import hudson.model.Cause

// Jenkins起動時にseed jobを実行
println "Running startup trigger script..."

def jenkins = Jenkins.getInstance()

// Jenkins が完全に起動するまで少し待機
Thread.sleep(10000)

try {
    def job = jenkins.getItemByFullName('my-project-seed-job')
    if (job != null) {
        println "Found my-project-seed-job, triggering build..."
        job.scheduleBuild(0, new Cause.UserIdCause("startup-script"))
        println "Seed job triggered successfully on startup"
    } else {
        println "my-project-seed-job not found, it may not be created yet"
    }
} catch (Exception e) {
    println "Error triggering seed job: ${e.getMessage()}"
} 
