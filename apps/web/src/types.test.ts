import { describe, it, expect } from "vitest";
import { money, percent, position } from "./types";
describe("account display units", () => {
  it("displays integer tenths without confusing pounds and millions", () => {
    expect(money(3)).toBe("£0.3m");
    expect(money(156)).toBe("£15.6m");
  });
  it("labels probabilities and positions consistently", () => {
    expect(percent(0.755)).toBe("76%");
    expect(position(1)).toBe("GK");
  });
});
