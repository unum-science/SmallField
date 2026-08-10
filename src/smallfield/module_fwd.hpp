/**
 *  @file module_fwd.hpp
 *  @brief Forward-declares every `smallfield` sub-namespace.
 *
 *  Lets headers that only need to name a namespace, not its contents,
 *  avoid pulling in the headers that populate it.
 */

#pragma once

namespace smallfield {
namespace api {}
namespace cuda {}
namespace mmff94 {}
namespace parameters {}
namespace preprocess {}
namespace types {}
} // namespace smallfield
