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
    page.getByRole("heading", { name: "Sources and model validation" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: "Learning transfer value across four weeks",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Role-aware minutes" }),
  ).toBeVisible();
  await expect(page.getByText("Temporal neural fusion", { exact: true })).toHaveCount(0);
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
  await expect(page.getByText('Published-signal ensemble in primary policy:',{exact:false})).toContainText('comparison only');
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
  await expect(page.getByRole('heading', {name:/(?:Production|Reference) pipeline · continuous account/})).toBeVisible();
  await expect(page.getByRole('heading', {name:'Future-fixture forecasting'})).toBeVisible();
  const expert = await (await page.request.get('/api/horizon/expert')).json();
  expect(expert.active.features.length).toBe(138);
  expect(expert.active.policy_gate_passed).toBe(true);
  expect(expert.forecast_validation.forecast_gate_passed).toBe(true);
  expect(expert.distribution_validation.passed).toBe(true);
  expect(expert.policy_validation.verification.sealed_deadlines).toBe(19);
  expect(expert.current_diagnostic.diagnostic_only).toBe(true);
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

test("chip calendar renders all four chip types, retained chips and date ledgers", async ({page}) => {
  await page.route('**/api/dashboard*', async route => {
    const response = await route.fetch();
    const body = await response.json();
    const chips = ['wildcard', 'freehit', 'bboost', '3xc'];
    body.chip_counterfactual = {
      base_plan_id: body.plan.id,
      dates_evaluated: 4,
      scope: 'Each forced-date policy jointly optimizes the other chips.',
      comparisons: chips.map((chip, i) => ({chip, window: [1, 19], chosen_event: i ? 12 : null,
        whole_policy_mean_delta: 1.25, paired_mc_interval_95: [0.5, 2], counterfactual_solver_gap: 0.02})),
      calendar: chips.map((chip, i) => ({chip, event: 10 + i, plan_id: body.plan.id,
        mean: 600 + i, delta_vs_selected: i - 1, solver: {gap: 0.02}})),
    };
    await route.fulfill({response, json: body});
  });
  await openWorkspace(page);
  await navigate(page, 'Season planner');
  const panel = page.locator('section').filter({has: page.getByRole('heading', {name: 'Chip value across the whole plan'})});
  await expect(panel.getByText('Wildcard · retained')).toBeVisible();
  await panel.getByText('Compare every feasible chip date (4)').click();
  for (const name of ['Wildcard', 'Free Hit', 'Bench Boost', 'Triple Captain']) {
    await expect(panel.getByRole('cell', {name, exact: true})).toBeVisible();
  }
  await expect(panel.getByRole('link', {name: 'Download', exact: true})).toHaveCount(4);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false);
});

test("withheld paid hits show search uncertainty, weekly payback and forecast disagreement", async ({page}) => {
  await page.route('**/api/dashboard*',async route=>{
    const response=await route.fetch(),body=await response.json();
    body.plan.ledger[0].hit=0;
    body.policy_selection={selection_version:'resource-evidence-v3',selection:{selected_plan_id:body.plan.id,actual_ev_sacrifice:1.73},immediate_hit_evidence:{candidate_plan_id:'raw-paid',no_hit_plan_id:body.plan.id,net_mean_gain:1.73,no_hit_search_allowance:9.31,hit_supported_beyond_search_error:false,paired_mc_interval_95:[1.45,2.01],reason:'Paid-transfer advantage is unresolved after the no-hit search allowance',weekly_tradeoff:[{event:10,candidate_net:44.33,alternative_net:53.34,cumulative_net_advantage:-9.01}],source_forecast_review:[{player_id:8,name:'Calafiori',fixtures:['HUL (H)'],model_points:3.10,model_minutes:58.7,model_start_probability:0.74,official_next_points:5.5,official_status:'a',official_availability:1,season_starts:8,completed_club_fixtures:9}]}};
    body.policy_selection.immediate_chip_evidence={chip:'freehit',candidate_plan_id:'raw-chip',alternative_plan_id:body.plan.id,net_mean_gain:1.19,search_allowance:11.72,supported_beyond_search_error:false,reason:'Chip retained while ordinary-transfer search remains unresolved'};
    await route.fulfill({response,json:body});
  });
  await openWorkspace(page);
  await navigate(page, 'Season planner');
  await expect(page.getByRole('heading',{name:'Free Hit retained'})).toBeVisible();
  const panel=page.locator('section').filter({has:page.getByRole('heading',{name:'Paid transfer withheld'})});
  await expect(panel).toContainText('9.31');
  await panel.getByText('See when the paid hit would repay its cost').click();
  await expect(panel.getByRole('cell',{name:'-9.01',exact:true})).toBeVisible();
  await panel.getByText('Inspect the player forecasts behind the paid proposal').click();
  await expect(panel.getByRole('cell',{name:'HUL (H)',exact:true})).toBeVisible();
  await expect(panel.getByRole('link',{name:'Download raw paid-hit proposal'})).toHaveAttribute('href','/api/export/raw-paid/json');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1)).toBe(false);
});
