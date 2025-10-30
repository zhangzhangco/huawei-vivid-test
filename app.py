#!/usr/bin/env python3
"""
HDR色调映射专利可视化工具 - Hugging Face Spaces 版本
"""

import sys
import os
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(current_dir, 'src'))


# 创建应用
print("🚀 启动HDR色调映射工具...")

try:
    # 导入完整版本
    from gradio_app import GradioInterface
    print("✅ 导入完整版本成功")

    interface = GradioInterface()
    app = interface.create_interface()
    print("✅ 完整应用创建成功")

except ImportError as e:
    error_msg = f"❌ 导入失败: {str(e)}"
    print(error_msg)
    print("\n🔧 可能的解决方案:")
    print("1. 检查src目录是否存在且包含必要文件")
    print("2. 确保所有依赖包已正确安装")
    print("3. 检查Python路径配置")
    raise SystemExit(error_msg)

except Exception as e:
    error_msg = f"❌ 应用创建失败: {str(e)}"
    print(error_msg)
    print("\n🔧 详细错误信息:")
    import traceback
    traceback.print_exc()
    print("\n💡 请检查:")
    print("1. 所有依赖模块是否正确实现")
    print("2. Gradio版本是否兼容")
    print("3. 系统环境是否满足要求")
    raise SystemExit(error_msg)

print(f"📱 应用类型: {type(app)}")

# 启用Gradio队列以支持并发处理
if hasattr(app, 'queue'):
    app.queue()
    print("✅ Gradio队列已启用")

# Hugging Face Spaces 会自动识别这个变量
if __name__ == "__main__":
    try:
        # 尝试获取API信息以验证配置
        app.get_api_info()
        print("✅ API验证成功")
    except Exception as e:
        print(f"⚠️ API验证失败: {e}")
        print("继续启动应用...")
        
        # 修补Gradio的API信息生成问题，确保后续不会再次触发相同异常
        try:
            from gradio import components, utils
            from gradio_client import utils as client_utils

            def _coerce_schema(info):
                if isinstance(info, dict):
                    return info
                if isinstance(info, bool):
                    return {"type": "boolean", "const": info}
                if info is None:
                    return {"type": "any"}
                return {"type": str(type(info).__name__)}

            def _custom_get_api_info(all_endpoints: bool = False):
                config = app.config
                api_info = {"named_endpoints": {}, "unnamed_endpoints": {}}

                for fn in app.fns.values():
                    if not fn.fn or fn.api_name is False:
                        continue
                    if not all_endpoints and not fn.show_api:
                        continue

                    dependency_info = {"parameters": [], "returns": [], "show_api": fn.show_api}
                    fn_info = utils.get_function_params(fn.fn)  # type: ignore
                    skip_endpoint = False

                    inputs = fn.inputs
                    for index, input_block in enumerate(inputs):
                        component = next(
                            (comp for comp in config["components"] if comp["id"] == input_block._id),
                            None,
                        )
                        if component is None:
                            skip_endpoint = True
                            break
                        if app.blocks[component["id"]].skip_api:  # type: ignore[index]
                            continue

                        label = component["props"].get("label", f"parameter_{input_block._id}")
                        comp_obj = app.get_component(component["id"])
                        if not isinstance(comp_obj, components.Component):
                            skip_endpoint = True
                            break
                        info = _coerce_schema(component.get("api_info"))
                        example = comp_obj.example_inputs()
                        try:
                            python_type = client_utils.json_schema_to_python_type(info)
                        except Exception:
                            python_type = "Any"

                        if (
                            fn.fn
                            and index < len(fn_info)
                            and fn_info[index][0] not in ["api_name", "fn_index", "result_callbacks"]
                        ):
                            parameter_name = fn_info[index][0]
                        else:
                            parameter_name = f"param_{index}"

                        if component["props"].get("value") is not None:
                            parameter_has_default = True
                            parameter_default = component["props"]["value"]
                        elif (
                            fn.fn
                            and index < len(fn_info)
                            and fn_info[index][1]
                            and fn_info[index][2] is None
                        ):
                            parameter_has_default = True
                            parameter_default = None
                        else:
                            parameter_has_default = False
                            parameter_default = None

                        component_name = component["props"].get("name") or component["type"]

                        dependency_info["parameters"].append(
                            {
                                "label": label,
                                "parameter_name": parameter_name,
                                "parameter_has_default": parameter_has_default,
                                "parameter_default": parameter_default,
                                "type": info,
                                "python_type": {
                                    "type": python_type,
                                    "description": info.get("description", ""),
                                },
                                "component": component_name.capitalize(),
                                "example_input": example,
                            }
                        )

                    outputs = fn.outputs
                    for output_block in outputs:
                        component = next(
                            (comp for comp in config["components"] if comp["id"] == output_block._id),
                            None,
                        )
                        if component is None:
                            skip_endpoint = True
                            break
                        if app.blocks[component["id"]].skip_api:  # type: ignore[index]
                            continue

                        label = component["props"].get("label", f"value_{output_block._id}")
                        comp_obj = app.get_component(component["id"])
                        if not isinstance(comp_obj, components.Component):
                            skip_endpoint = True
                            break
                        info = _coerce_schema(component.get("api_info"))
                        example = comp_obj.example_inputs()
                        try:
                            python_type = client_utils.json_schema_to_python_type(info)
                        except Exception:
                            python_type = "Any"

                        component_name = component["props"].get("name") or component["type"]

                        dependency_info["returns"].append(
                            {
                                "label": label,
                                "type": info,
                                "python_type": {
                                    "type": python_type,
                                    "description": info.get("description", ""),
                                },
                                "component": component_name.capitalize(),
                                "example_input": example,
                            }
                        )

                    if not skip_endpoint:
                        api_info["named_endpoints"][f"/{fn.api_name}"] = dependency_info

                return api_info

            app.get_api_info = _custom_get_api_info  # type: ignore[assignment]
            app.api_info = _custom_get_api_info()
            print("✅ 已禁用API信息生成")
        except Exception as patch_error:
            print(f"⚠️ API补丁失败: {patch_error}")

    app.launch(
        share=True, 
        show_api=False,
        show_error=True,
        quiet=False
    )
