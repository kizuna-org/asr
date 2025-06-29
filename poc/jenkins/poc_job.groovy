multibranchPipelineJob('my-poc-repo') {
    description('Multibranch pipeline for POC repository')

    branchSources {
        git {
            id('ea86f402-26ae-44a2-a857-c0ccc5c30a4d')
            remote('http://gitea:3000/poc_user/my-poc-repo.git')
            credentialsId('gitea-credentials')
            traits {
                branchDiscoveryTrait()
            }
        }
    }

    orphanedItemStrategy {
        discardOldItems {
            daysToKeep(-1)
            numToKeep(-1)
        }
    }

    factory {
        workflowBranchProjectFactory {
            scriptPath('Jenkinsfile')
        }
    }
}
