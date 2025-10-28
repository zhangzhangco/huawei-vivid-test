#!/usr/bin/env python3
"""
验证测试运行器
提供统一的测试执行入口和配置管理
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_validation_framework import AutomatedRegressionTestSuite
from test_regression_suite import run_comprehensive_regression_tests
from golden_test_data import create_golden_test_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HDR色调映射验证测试运行器")
    parser.add_argument("--test-type", choices=["all", "golden", "monotonicity", "hysteresis", "consistency", "regression"], 
                       default="all", help="测试类型")
    parser.add_argument("--output-dir", default="test_results", help="输出目录")
    parser.add_argument("--generate-data", action="store_true", help="生成金标测试数据")
    parser.add_argument("--report-format", choices=["markdown", "json", "both"], default="both", help="报告格式")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成测试数据
    if args.generate_data:
        print("生成金标测试数据...")
        create_golden_test_data()
        
    # 运行测试
    if args.test_type == "regression":
        success = run_comprehensive_regression_tests()
        return 0 if success else 1
    else:
        test_suite = AutomatedRegressionTestSuite()
        results = test_suite.run_full_regression_suite()
        
        # 生成报告
        if args.report_format in ["markdown", "both"]:
            report_file = os.path.join(args.output_dir, "validation_report.md")
            test_suite.generate_test_report(results, report_file)
            print(f"Markdown报告已生成: {report_file}")
            
        if args.report_format in ["json", "both"]:
            json_file = os.path.join(args.output_dir, "validation_results.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"JSON结果已保存: {json_file}")
            
        # 输出摘要
        summary = results["test_summary"]
        print(f"\n测试完成: {summary['passed_tests']}/{summary['total_tests']} 通过")
        
        return 0 if summary["failed_tests"] == 0 else 1


if __name__ == "__main__":
    exit(main())