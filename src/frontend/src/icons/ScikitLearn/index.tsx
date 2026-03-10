import React, { forwardRef } from "react";
import SvgScikitLearnLogo from "./ScikitLearnLogo";

export const ScikitLearnIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  return <SvgScikitLearnLogo ref={ref} {...props} />;
});
