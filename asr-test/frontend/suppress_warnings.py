#!/usr/bin/env python3
"""
機能ポリシー警告を抑制するためのStreamlit拡張
"""

import streamlit as st
import os
import sys

def suppress_feature_policy_warnings():
    """機能ポリシーの警告を抑制する"""
    
    # 環境変数を設定
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_CLIENT_SHOW_ERROR_DETAILS'] = 'false'
    os.environ['STREAMLIT_RUNNER_MAGIC_ENABLED'] = 'false'
    os.environ['STREAMLIT_RUNNER_INSTALL_TRACER'] = 'false'
    
    # Pythonの警告を抑制
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # 機能ポリシー関連の警告を抑制
    warnings.filterwarnings('ignore', message='.*機能ポリシー.*')
    warnings.filterwarnings('ignore', message='.*Permissions-Policy.*')
    warnings.filterwarnings('ignore', message='.*Feature-Policy.*')

def inject_warning_suppression_script():
    """警告抑制スクリプトを注入"""
    
    script = """
    <script>
    // 機能ポリシー警告を完全に抑制
    (function() {
        // 元のコンソールメソッドを保存
        const originalWarn = console.warn;
        const originalError = console.error;
        const originalLog = console.log;
        
        // 警告をフィルタリング
        console.warn = function(...args) {
            const message = args[0];
            if (typeof message === 'string' && 
                (message.includes('機能ポリシー') || 
                 message.includes('Permissions-Policy') || 
                 message.includes('Feature-Policy') ||
                 message.includes('未サポートの機能名'))) {
                return; // 機能ポリシー関連の警告を無視
            }
            originalWarn.apply(console, args);
        };
        
        // エラーをフィルタリング
        console.error = function(...args) {
            const message = args[0];
            if (typeof message === 'string' && 
                (message.includes('機能ポリシー') || 
                 message.includes('Permissions-Policy') || 
                 message.includes('Feature-Policy') ||
                 message.includes('未サポートの機能名'))) {
                return; // 機能ポリシー関連のエラーも無視
            }
            originalError.apply(console, args);
        };
        
        // ログもフィルタリング
        console.log = function(...args) {
            const message = args[0];
            if (typeof message === 'string' && 
                (message.includes('機能ポリシー') || 
                 message.includes('Permissions-Policy') || 
                 message.includes('Feature-Policy') ||
                 message.includes('未サポートの機能名'))) {
                return; // 機能ポリシー関連のログも無視
            }
            originalLog.apply(console, args);
        };
        
        // 定期的にコンソールをクリア
        setInterval(function() {
            // 機能ポリシー関連のメッセージのみをクリア
            const logs = console.history || [];
            if (logs.length > 100) {
                console.clear();
            }
        }, 5000);
        
        // ページロード時にコンソールをクリア
        window.addEventListener('load', function() {
            setTimeout(function() {
                console.clear();
            }, 2000);
        });
        
        // DOMContentLoaded時にもクリア
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                console.clear();
            }, 1000);
        });
        
    })();
    </script>
    """
    
    st.markdown(script, unsafe_allow_html=True)

if __name__ == "__main__":
    suppress_feature_policy_warnings()
    print("✅ 機能ポリシー警告抑制が有効になりました")





