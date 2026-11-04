import { test, expect } from "@playwright/test";

async function openWorkspace(page: import("@playwright/test").Page) {
  await Promise.all([
    page.waitForResponse(response => response.url().includes("/api/dashboard") && response.status() === 200),
    page.goto("/"),
  ]);
}

async function navigate(page: import("@playwright/test").Page, label: string) {
  const mobileMenu = page.getByRole("button", { name: "Toggle navigation" });
  if (await mobileMenu.isVisible()) await mobileMenu.click();
  await page.getByRole("button", { name: label, exact: true }).click();
}

test("recommendation, legal XI, full market and planner exports", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openWorkspace(page);
  await expect(
    page.getByRole("heading", { name: "The gameweek, in focus." }),
  ).toBeVisible();
  await expect(
    page.getByText("THE RECOMMENDATION", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".pitch-player")).toHaveCount(11);
  await expect(page.locator(".pitch .captain:not(.vice)")).toHaveCount(1);
  await navigate(page, "Player market");
  const market = await (await page.request.get("/api/dashboard")).json();
  await expect(page.locator("tbody tr")).toHaveCount(market.market.players.length);
  await page.getByRole("textbox", { name: "Search players" }).fill("Haaland");
  await expect(page.locator("tbody tr")).toHaveCount(1);
  await expect(page.locator("tbody")).toContainText("Haaland");
  await navigate(page, "Season planner");
  await expect(
    page.getByRole("heading", { name: "The full gameweek ledger" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "CSV", exact: true }),
  ).toHaveAttribute("href", /\/api\/export\/.+\/csv/);
  await expect(page.locator(".calendar-cell")).toHaveCount(38);
  await page.getByRole("button", { name: /GW 12/ }).click();
  await expect(page.getByRole("combobox", { name: "Gameweek" })).toHaveValue(
    "12",
  );
  expect(errors).toEqual([]);
  await page.screenshot({
    path: `test-results/${test.info().project.name}-overview.png`,
    fullPage: true,
  });
});

test("squad editor validates and transfer checker rejects unaffordable moves", async ({
  page,
}) => {
  await openWorkspace(page);
  await navigate(page, "My squad");
  await expect(page.locator("tbody tr")).toHaveCount(15);
  await page
    .getByRole("button", { name: "Edit account state", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toBeVisible();
  const body = JSON.parse(
    await page
      .getByRole("textbox", { name: "Account state JSON" })
      .inputValue(),
  );
  body.bank = -1;
  await page
    .getByRole("textbox", { name: "Account state JSON" })
    .fill(JSON.stringify(body));
  await page.getByRole("button", { name: "Validate & save account" }).click();
  await expect(page.getByRole("alert")).toBeVisible();
  await page.getByRole("button", { name: "Dismiss error" }).click();
  if (await page.getByRole("dialog").isVisible())
    await page.getByRole("button", { name: "Close editor" }).click();
  await navigate(page, "Model & data");
  await expect(
    page.getByRole("heading", { name: "Models earn their place." }),
  ).toBeVisible();
  await expect(
    page.getByText("Completed-match statistical replay", { exact: false }),
  ).toBeVisible();
});

test("manager behavior, source fusion and transfer learning are visible", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openWorkspace(page);
  await navigate(page, "Rivals");
  await expect(
    page.getByRole("heading", { name: "Top-1,000 manager behavior" }),
  ).toBeVisible();
  await expect(page.getByText(/1000 actual managers/)).toBeVisible();
  await expect(
    page.getByText("Learned-count scenario", { exact: true }),
  ).toBeVisible();
  await navigate(page, "Model & data");
  await expect(
    page.getByRole("heading", { name: "Sources and neural validation" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: "Learning transfer value across four weeks",
    }),
  ).toBeVisible();
  await expect(
    page.getByText("Temporal neural fusion", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText(/\d+ held-out account decisions/)).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth + 1,
  );
  expect(overflow).toBe(false);
  expect(errors).toEqual([]);
});

test("connected strategy, timing and population controls render without errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await openWorkspace(page);
  await navigate(page, "Season planner");
  await expect(page.getByRole("heading", {name:"Transfer timing and affordability"})).toBeVisible();
  await expect(page.getByRole("button", {name:"Analyze timing",exact:true})).toBeVisible();
  await navigate(page, "Rivals");
  await expect(page.getByRole("heading", {name:"Population and rank scenarios"})).toBeVisible();
  await navigate(page, "Model & data");
  await expect(page.getByRole("heading", {name:"Learning decisions, not just points"})).toBeVisible();
  await expect(page.getByRole("button", {name:"Refine multiweek frontier",exact:true})).toBeVisible();
  await expect(page.getByRole("button", {name:"Refresh and recompute",exact:true})).toBeVisible();
  await expect(page.getByRole("button", {name:"Compare football worlds",exact:true})).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false);
  expect(errors).toEqual([]);
});


test("completed weekly replay and football integration are visible",async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await openWorkspace(page);await navigate(page,'Model & data');
  await expect(page.getByRole('heading',{name:'Football models and current coverage'})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Historical weekly teams · refitted reference'})).toBeVisible();
  const response=await page.request.get('/api/replay/weekly');expect(response.status()).toBe(200);
  const replay=await response.json();expect(replay.weeks.map((w:{event:number})=>w.event)).toEqual([1,2,3,4,5,6,7,8,9]);
  expect(replay.total_points).toBeGreaterThan(0);
  await expect(page.getByRole('heading',{name:'Published-signal fusion · weekly experiment'})).toBeVisible();
  const fusion=await (await page.request.get('/api/consensus/model')).json();
  expect(fusion.acceptance.passed).toBe(true);
  await expect(page.getByRole('link',{name:'Download full experiment and frozen teams'})).toHaveAttribute('href','/api/consensus/model');
  await expect(page.getByText('Primary multiweek model:',{exact:false})).toContainText('unchanged');
  await navigate(page,'Season planner');
  await page.getByRole('combobox',{name:'Scenario',exact:true}).selectOption('fixture');
  await expect(page.getByRole('combobox',{name:'Fixture',exact:true})).toBeVisible();
  expect(errors).toEqual([]);
});

test("stale recommendations cannot display old transfers as current actions",async({page})=>{
  await page.route('**/api/dashboard*',async route=>{
    const response=await route.fetch();const body=await response.json();
    if(body.plan)body.plan.stale=true;
    await route.fulfill({response,json:body});
  });
  await openWorkspace(page);
  await expect(page.getByText('This plan uses an older account, forecast or market version.',{exact:false})).toBeVisible();
  await expect(page.locator('.transfer-pair')).toHaveCount(0);
  await expect(page.locator('.pitch-player')).toHaveCount(0);
});

test("integrated replay proofs and synchronized future accounts are accessible", async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await openWorkspace(page);
  await navigate(page, 'Model & data');
  await expect(page.getByRole('heading', {name:'Production pipeline · continuous account'})).toBeVisible();
  await expect(page.getByRole('heading', {name:'Prospective continuous accounts'})).toBeVisible();
  const replayResponse = await page.request.get('/api/replay/pipeline');
  expect(replayResponse.status()).toBe(200);
  const replay = await replayResponse.json();
  expect(replay.production_budget).toBe(true);
  expect(replay.independent_verification.validated_actions).toBe(84);
  expect(replay.independent_verification.account_continuity).toBe('passed');
  const first = replay.weeks.find((r:{season:string;event:number;policy:string}) =>
    r.season === '2026/27' && r.event === 1 && r.policy === 'primary_multiweek');
  const proofResponse = await page.request.get('/api/replay/pipeline/decision/2026-27/1');
  expect(proofResponse.status()).toBe(200);
  const proof = await proofResponse.json();
  expect(proof.actions[0].action).toEqual(first.action);
  expect(proof.registered_players.length).toBeGreaterThanOrEqual(15);
  expect(proof.model.fit_evidence.calibration_max_group < '2026/27-GW01').toBe(true);
  const shadow = await (await page.request.get('/api/prospective/accounts')).json();
  expect(shadow.shadow_cohort.real_account_unchanged).toBe(true);
  expect(Object.keys(shadow.shadow_cohort.actions).length).toBe(3);
  await expect(page.getByRole('link', {name:'Download frozen decisions and results'})).toHaveAttribute('href','/api/prospective/accounts');
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false);
  expect(errors).toEqual([]);
});
