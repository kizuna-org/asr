#!/bin/bash
# extract_jenkins_config.sh
# Jenkins Web UI設定を抽出してJob DSL変換を支援するスクリプト

set -e

# デフォルト設定
JENKINS_URL="${JENKINS_URL:-http://localhost:8080}"
JENKINS_USER="${JENKINS_USER:-admin}"
JENKINS_PASS="${JENKINS_PASS:-admin}"
OUTPUT_DIR="./extracted_configs"

# ヘルプメッセージ
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] JOB_NAME

Jenkins Web UI設定を抽出してJob DSL変換を支援するスクリプト

OPTIONS:
    -u, --url URL       Jenkins URL (default: http://localhost:8080)
    -U, --user USER     Jenkins username (default: admin)
    -p, --pass PASS     Jenkins password (default: admin)
    -o, --output DIR    Output directory (default: ./extracted_configs)
    -a, --all           Extract all jobs
    -h, --help          Show this help message

EXAMPLES:
    $0 my-pipeline-job                  # 特定のジョブ設定を抽出
    $0 -a                               # すべてのジョブ設定を抽出
    $0 -u http://jenkins:8080 my-job    # 異なるURLを指定

EOF
}

# オプション解析
extract_all=false
job_name=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            JENKINS_URL="$2"
            shift 2
            ;;
        -U|--user)
            JENKINS_USER="$2"
            shift 2
            ;;
        -p|--pass)
            JENKINS_PASS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -a|--all)
            extract_all=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [ -z "$job_name" ]; then
                job_name="$1"
            else
                echo "Error: Unknown option $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# パラメータチェック
if [ "$extract_all" = false ] && [ -z "$job_name" ]; then
    echo "Error: Job name is required or use -a to extract all jobs"
    show_help
    exit 1
fi

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# Jenkins接続テスト
echo "Testing Jenkins connection..."
if ! curl -s -u "${JENKINS_USER}:${JENKINS_PASS}" "${JENKINS_URL}/api/json" > /dev/null; then
    echo "Error: Cannot connect to Jenkins at ${JENKINS_URL}"
    echo "Please check URL, username, and password"
    exit 1
fi
echo "✅ Jenkins connection successful"

# ジョブ一覧取得関数
get_all_jobs() {
    curl -s -u "${JENKINS_USER}:${JENKINS_PASS}" \
        "${JENKINS_URL}/api/json?tree=jobs[name,_class]" | \
        jq -r '.jobs[] | select(._class | contains("Pipeline")) | .name'
}

# 単一ジョブ設定抽出関数
extract_job_config() {
    local job="$1"
    local encoded_job=$(printf '%s\n' "$job" | jq -sRr @uri)
    local config_file="${OUTPUT_DIR}/${job}_config.xml"
    local summary_file="${OUTPUT_DIR}/${job}_summary.txt"
    
    echo "Extracting configuration for job: $job"
    
    # XML設定を取得
    if curl -s -u "${JENKINS_USER}:${JENKINS_PASS}" \
        "${JENKINS_URL}/job/${encoded_job}/config.xml" \
        -o "$config_file"; then
        echo "✅ Config saved to: $config_file"
    else
        echo "❌ Failed to extract config for job: $job"
        return 1
    fi
    
    # ジョブ情報サマリーを作成
    {
        echo "=== Job Configuration Summary ==="
        echo "Job Name: $job"
        echo "Extracted: $(date)"
        echo "Jenkins URL: $JENKINS_URL"
        echo ""
        
        # ジョブタイプを判定
        if grep -q "org.jenkinsci.plugins.workflow.multibranch.WorkflowMultiBranchProject" "$config_file"; then
            echo "Job Type: Multibranch Pipeline"
            echo ""
            echo "=== Branch Sources ==="
            grep -A 20 "<sources>" "$config_file" | head -20 || echo "Could not extract branch sources"
            
        elif grep -q "org.jenkinsci.plugins.workflow.job.WorkflowJob" "$config_file"; then
            echo "Job Type: Pipeline"
            echo ""
            echo "=== SCM Configuration ==="
            grep -A 10 "<scm class=" "$config_file" | head -10 || echo "Could not extract SCM config"
            
        elif grep -q "hudson.model.FreeStyleProject" "$config_file"; then
            echo "Job Type: Freestyle"
            echo ""
            echo "=== Build Steps ==="
            grep -A 10 "<builders>" "$config_file" | head -10 || echo "Could not extract build steps"
        fi
        
        echo ""
        echo "=== Triggers ==="
        grep -A 5 "<triggers>" "$config_file" | head -10 || echo "No triggers configured"
        
        echo ""
        echo "=== Parameters ==="
        grep -A 5 "<parameterDefinitions>" "$config_file" | head -10 || echo "No parameters configured"
        
    } > "$summary_file"
    
    echo "📄 Summary saved to: $summary_file"
}

# Job DSL変換サンプル生成関数
generate_job_dsl_template() {
    local job="$1"
    local config_file="${OUTPUT_DIR}/${job}_config.xml"
    local dsl_file="${OUTPUT_DIR}/${job}_job_dsl_template.groovy"
    
    if [ ! -f "$config_file" ]; then
        echo "Config file not found for $job"
        return 1
    fi
    
    echo "Generating Job DSL template for: $job"
    
    {
        echo "// Job DSL Template for: $job"
        echo "// Generated: $(date)"
        echo "// NOTE: This is a template. Please review and customize as needed."
        echo ""
        
        if grep -q "WorkflowMultiBranchProject" "$config_file"; then
            cat << 'EOF'
multibranchPipelineJob('REPLACE_JOB_NAME') {
    displayName('REPLACE_DISPLAY_NAME')
    description('REPLACE_DESCRIPTION')
    
    branchSources {
        git {
            id('git-source')
            remote('REPLACE_REPOSITORY_URL')
            credentialsId('REPLACE_CREDENTIALS_ID')
            traits {
                gitBranchDiscovery()
                // gitTagDiscovery()
            }
        }
    }
    
    factory {
        workflowBranchProjectFactory {
            scriptPath('Jenkinsfile')
        }
    }
    
    triggers {
        periodic(5) // minutes
    }
    
    orphanedItemStrategy {
        discardOldItems {
            daysToKeep(7)
            numToKeep(20)
        }
    }
}
EOF
        elif grep -q "WorkflowJob" "$config_file"; then
            cat << 'EOF'
pipelineJob('REPLACE_JOB_NAME') {
    displayName('REPLACE_DISPLAY_NAME')
    description('REPLACE_DESCRIPTION')
    
    definition {
        cpsScm {
            scm {
                git {
                    remote {
                        url('REPLACE_REPOSITORY_URL')
                        credentials('REPLACE_CREDENTIALS_ID')
                    }
                    branch('*/main')
                }
            }
            scriptPath('Jenkinsfile')
        }
    }
    
    triggers {
        scm('H/15 * * * *')
    }
    
    logRotator {
        numToKeep(10)
        daysToKeep(30)
    }
}
EOF
        else
            echo "// Unsupported job type or freestyle job"
            echo "// Please refer to Job DSL documentation for freestyle job DSL"
        fi
        
    } > "$dsl_file"
    
    echo "📝 Job DSL template saved to: $dsl_file"
}

# メイン処理
echo "=== Jenkins Job Configuration Extractor ==="
echo "Jenkins URL: $JENKINS_URL"
echo "Output Directory: $OUTPUT_DIR"
echo ""

if [ "$extract_all" = true ]; then
    echo "Extracting all pipeline jobs..."
    jobs=$(get_all_jobs)
    
    if [ -z "$jobs" ]; then
        echo "No pipeline jobs found"
        exit 0
    fi
    
    job_count=0
    while IFS= read -r job; do
        if [ -n "$job" ]; then
            extract_job_config "$job"
            generate_job_dsl_template "$job"
            echo ""
            ((job_count++))
        fi
    done <<< "$jobs"
    
    echo "✅ Extracted $job_count jobs successfully"
else
    extract_job_config "$job_name"
    generate_job_dsl_template "$job_name"
    echo ""
    echo "✅ Extraction completed for job: $job_name"
fi

echo ""
echo "=== Next Steps ==="
echo "1. Review the extracted XML configuration files"
echo "2. Customize the generated Job DSL templates"
echo "3. Add the Job DSL script to your casc.yaml"
echo "4. Test the Job DSL script in Jenkins"
echo ""
echo "Files saved in: $OUTPUT_DIR" 
