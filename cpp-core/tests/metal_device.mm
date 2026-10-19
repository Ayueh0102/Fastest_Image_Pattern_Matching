// SPDX-License-Identifier: BSD-2-Clause
#include "metal_device.hpp"
#import <Metal/Metal.h>

bool fpm_test::metalDeviceAvailable()
{
    @autoreleasepool { return MTLCreateSystemDefaultDevice() != nil; }
}
