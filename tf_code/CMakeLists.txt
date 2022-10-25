set(binaryName "pico-tensorflow")
add_executable(${binaryName}
	main.cpp
	#conv_mnist_quant.h
	#conv_mnist_quant.cpp
)

pico_enable_stdio_usb(${binaryName} 1)
pico_enable_stdio_uart(${binaryName} 1)

target_link_libraries(${binaryName}
	pico_stdlib
	hardware_spi
	pico-tflmicro
)

pico_add_extra_outputs(${binaryName})
