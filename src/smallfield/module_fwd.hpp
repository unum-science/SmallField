/**
 *  @file src/smallfield/module_fwd.hpp
 *  @brief Forward-declares the `smallfield` sub-namespaces.
 *
 *  Lets a translation unit name a namespace, rather than its contents, without pulling in the headers
 *  that populate it. `tests` is here for one specific reason: 13 test translation units open
 *  `smallfield::tests` without including `test_helpers.hpp`, and two of those alias it at file scope,
 *  which is only legal once the namespace has been declared. `bench` and `cli` stay out because each
 *  is named only after its own headers declare it.
 */

#pragma once

namespace smallfield {
namespace api {}
namespace cuda {}
namespace mmff94 {}
namespace parameters {}
namespace postprocess {}
namespace preprocess {}
namespace tests {}
namespace types {}
} // namespace smallfield
