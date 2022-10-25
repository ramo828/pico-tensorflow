# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# compile ASM with /usr/bin/arm-none-eabi-gcc
# compile C with /usr/bin/arm-none-eabi-gcc
# compile CXX with /usr/bin/arm-none-eabi-g++
ASM_DEFINES = -DCMSIS_NN=1 -DCOMPILE_DEFINITIONS -DPICO_BIT_OPS_PICO=1 -DPICO_BOARD=\"pico\" -DPICO_BUILD=1 -DPICO_CMAKE_BUILD_TYPE=\"Release\" -DPICO_COPY_TO_RAM=0 -DPICO_CXX_ENABLE_EXCEPTIONS=0 -DPICO_DIVIDER_HARDWARE=1 -DPICO_DOUBLE_PICO=1 -DPICO_FLOAT_PICO=1 -DPICO_INT64_OPS_PICO=1 -DPICO_MEM_OPS_PICO=1 -DPICO_NO_FLASH=0 -DPICO_NO_HARDWARE=0 -DPICO_ON_DEVICE=1 -DPICO_PRINTF_PICO=1 -DPICO_STDIO_UART=1 -DPICO_USE_BLOCKED_RAM=0 -DTF_LITE_DISABLE_X86_NEON=1 -DTF_LITE_STATIC_MEMORY=1

ASM_INCLUDES = -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis/CMSIS/DSP/Include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/ruy -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/gemmlowp -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/kissfft -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/flatbuffers -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis/CMSIS/Core/Include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/flatbuffers/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis/CMSIS/NN/Include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_stdlib/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_gpio/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_base/include -I/home/ramo828/Belgeler/PICO/pico-mnist/generated/pico_base -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/boards/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_platform/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2040/hardware_regs/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_base/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2040/hardware_structs/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_claim/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_sync/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_uart/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_divider/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_time/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_timer/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_sync/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_util/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_runtime/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_clocks/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_resets/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_watchdog/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_xosc/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_pll/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_vreg/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_irq/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_printf/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_bootrom/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_bit_ops/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_divider/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_double/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_int64_ops/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_float/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_binary_info/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_stdio/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_stdio_uart/include

ASM_FLAGS = -march=armv6-m -mcpu=cortex-m0plus -mthumb -O3 -DNDEBUG -nostdlib -ffunction-sections -fdata-sections

C_DEFINES = -DCMSIS_NN=1 -DCOMPILE_DEFINITIONS -DPICO_BIT_OPS_PICO=1 -DPICO_BOARD=\"pico\" -DPICO_BUILD=1 -DPICO_CMAKE_BUILD_TYPE=\"Release\" -DPICO_COPY_TO_RAM=0 -DPICO_CXX_ENABLE_EXCEPTIONS=0 -DPICO_DIVIDER_HARDWARE=1 -DPICO_DOUBLE_PICO=1 -DPICO_FLOAT_PICO=1 -DPICO_INT64_OPS_PICO=1 -DPICO_MEM_OPS_PICO=1 -DPICO_NO_FLASH=0 -DPICO_NO_HARDWARE=0 -DPICO_ON_DEVICE=1 -DPICO_PRINTF_PICO=1 -DPICO_STDIO_UART=1 -DPICO_USE_BLOCKED_RAM=0 -DTF_LITE_DISABLE_X86_NEON=1 -DTF_LITE_STATIC_MEMORY=1

C_INCLUDES = -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis/CMSIS/DSP/Include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/ruy -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/gemmlowp -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/kissfft -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/flatbuffers -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis/CMSIS/Core/Include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/flatbuffers/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis/CMSIS/NN/Include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_stdlib/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_gpio/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_base/include -I/home/ramo828/Belgeler/PICO/pico-mnist/generated/pico_base -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/boards/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_platform/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2040/hardware_regs/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_base/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2040/hardware_structs/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_claim/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_sync/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_uart/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_divider/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_time/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_timer/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_sync/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_util/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_runtime/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_clocks/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_resets/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_watchdog/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_xosc/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_pll/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_vreg/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_irq/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_printf/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_bootrom/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_bit_ops/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_divider/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_double/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_int64_ops/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_float/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_binary_info/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_stdio/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_stdio_uart/include

C_FLAGS = -march=armv6-m -mcpu=cortex-m0plus -mthumb -O3 -DNDEBUG -nostdlib -ffunction-sections -fdata-sections -std=gnu11

CXX_DEFINES = -DCMSIS_NN=1 -DCOMPILE_DEFINITIONS -DPICO_BIT_OPS_PICO=1 -DPICO_BOARD=\"pico\" -DPICO_BUILD=1 -DPICO_CMAKE_BUILD_TYPE=\"Release\" -DPICO_COPY_TO_RAM=0 -DPICO_CXX_ENABLE_EXCEPTIONS=0 -DPICO_DIVIDER_HARDWARE=1 -DPICO_DOUBLE_PICO=1 -DPICO_FLOAT_PICO=1 -DPICO_INT64_OPS_PICO=1 -DPICO_MEM_OPS_PICO=1 -DPICO_NO_FLASH=0 -DPICO_NO_HARDWARE=0 -DPICO_ON_DEVICE=1 -DPICO_PRINTF_PICO=1 -DPICO_STDIO_UART=1 -DPICO_USE_BLOCKED_RAM=0 -DTF_LITE_DISABLE_X86_NEON=1 -DTF_LITE_STATIC_MEMORY=1

CXX_INCLUDES = -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis/CMSIS/DSP/Include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/ruy -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/gemmlowp -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/kissfft -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/flatbuffers -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis/CMSIS/Core/Include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/flatbuffers/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-tflmicro/src/third_party/cmsis/CMSIS/NN/Include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_stdlib/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_gpio/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_base/include -I/home/ramo828/Belgeler/PICO/pico-mnist/generated/pico_base -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/boards/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_platform/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2040/hardware_regs/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_base/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2040/hardware_structs/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_claim/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_sync/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_uart/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_divider/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_time/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_timer/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_sync/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_util/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_runtime/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_clocks/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_resets/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_watchdog/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_xosc/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_pll/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_vreg/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/hardware_irq/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_printf/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_bootrom/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_bit_ops/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_divider/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_double/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_int64_ops/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_float/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/common/pico_binary_info/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_stdio/include -I/home/ramo828/Belgeler/PICO/pico-mnist/pico-sdk/src/rp2_common/pico_stdio_uart/include

CXX_FLAGS = -march=armv6-m -mcpu=cortex-m0plus -mthumb -O3 -DNDEBUG -nostdlib -ffunction-sections -fdata-sections -fno-exceptions -fno-unwind-tables -fno-rtti -fno-use-cxa-atexit -std=gnu++17

