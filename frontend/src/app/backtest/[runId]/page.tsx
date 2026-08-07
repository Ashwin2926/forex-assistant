import RunDetail from "./RunDetail";

export default async function BacktestRunPage(props: PageProps<"/backtest/[runId]">) {
  const { runId } = await props.params;
  return <RunDetail runId={runId} />;
}
