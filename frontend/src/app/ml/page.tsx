"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, PROFILES, type MLPrediction, type MLTrainResult, type Profile } from "@/lib/types";

export default function MLPage() {
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainResult, setTrainResult] = useState<MLTrainResult | null>(null);

  const [runs, setRuns] = useState<MLTrainResult[]>([]);
  const [runsLoading, setRunsLoading] = useState(true);

  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [interval, setInterval_] = useState<string>("1h");
  const [profile, setProfile] = useState<Profile>("swing");
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<MLPrediction | null>(null);

  async function loadRuns() {
    setRunsLoading(true);
    try {
      setRuns(await api.listMLRuns(20));
    } catch {
      // non-critical section -- a failed load here shouldn't block the rest of the page
    } finally {
      setRunsLoading(false);
    }
  }

  useEffect(() => {
    loadRuns();
  }, []);

  async function handleTrain() {
    setTraining(true);
    setTrainError(null);
    try {
      const result = await api.trainMLModel();
      setTrainResult(result);
      await loadRuns();
    } catch (e) {
      setTrainError(e instanceof ApiError ? e.message : "Training failed.");
    } finally {
      setTraining(false);
    }
  }

  async function handlePredict() {
    setPredicting(true);
    setPredictError(null);
    setPrediction(null);
    try {
      setPrediction(await api.predictML(pair, interval, profile));
    } catch (e) {
      setPredictError(e instanceof ApiError ? e.message : "Prediction failed.");
    } finally {
      setPredicting(false);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">ML (v1)</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          A supervised classifier (logistic regression, <strong>not</strong> reinforcement
          learning) trained on every resolved live signal to predict hit vs. not-hit. Trains
          automatically every cron cycle on whatever has resolved so far — with the current
          sample size, treat every prediction as a rough, evolving estimate, not a validated
          edge. Doesn&apos;t affect the regular signal feed or consensus in any way; purely
          advisory.
        </p>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Train the model now</h2>
          <button onClick={handleTrain} disabled={training} className="btn-primary">
            {training ? "Training…" : "Train model"}
          </button>
        </div>
        {trainError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {trainError}
          </p>
        )}
        {trainResult && (
          <div className="mt-4 flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-2">
              <MLStatCard label="Train" accuracy={trainResult.train_accuracy} samples={trainResult.train_samples} accent="zinc" />
              <MLStatCard
                label="Test (out-of-sample)" accuracy={trainResult.test_accuracy} samples={trainResult.test_samples} accent="emerald"
                precision={trainResult.test_precision} recall={trainResult.test_recall}
              />
            </div>
            <div>
              <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">Feature coefficients</p>
              <p className="mt-1 text-xs text-zinc-400 dark:text-zinc-500">
                What the model actually weighted — positive pushes toward &quot;hit,&quot; negative toward &quot;not hit.&quot;
              </p>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 rounded-md bg-zinc-50 px-3 py-2 text-xs font-mono dark:bg-zinc-800">
                {Object.entries(trainResult.feature_coefficients).map(([name, value]) => (
                  <span key={name}>
                    {name}:{" "}
                    <span className={value >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}>
                      {value >= 0 ? "+" : ""}{value.toFixed(3)}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Predict a live signal</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Generates a fresh signal (same as the signal feed) and attaches the model&apos;s hit
          probability — this call doesn&apos;t get stored anywhere, it&apos;s a one-off check.
        </p>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Pair">
            <select value={pair} onChange={(e) => setPair(e.target.value)} className="select">
              {PAIRS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Interval">
            <select value={interval} onChange={(e) => setInterval_(e.target.value)} className="select">
              {INTERVALS.map((i) => <option key={i} value={i}>{i}</option>)}
            </select>
          </Field>
          <Field label="Profile">
            <select value={profile} onChange={(e) => setProfile(e.target.value as Profile)} className="select">
              {PROFILES.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <button onClick={handlePredict} disabled={predicting} className="btn-primary">
            {predicting ? "Predicting…" : "Predict"}
          </button>
        </div>
        {predictError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {predictError}
          </p>
        )}
        {prediction && (
          <div className="mt-4 rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
            <p className="text-lg font-semibold">
              {prediction.direction}
              {prediction.direction !== "HOLD" && ` at ${prediction.price_at_signal.toFixed(5)}`}
            </p>
            {prediction.target_price != null && prediction.stop_price != null && (
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Target {prediction.target_price.toFixed(5)} · Stop {prediction.stop_price.toFixed(5)}
              </p>
            )}
            <p className="mt-2 text-sm">
              ML hit probability:{" "}
              {prediction.ml_hit_probability != null ? (
                <span className="font-semibold">{(prediction.ml_hit_probability * 100).toFixed(1)}%</span>
              ) : (
                <span className="text-zinc-400">not enough resolved data yet</span>
              )}
            </p>
          </div>
        )}
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Training history</h2>
        {runsLoading && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {!runsLoading && runs.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">No training runs yet — train the model above.</p>
        )}
        {runs.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-sm">
              <thead className="bg-zinc-50 text-xs uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                <tr>
                  <th className="px-4 py-2">Train / Test accuracy</th>
                  <th className="px-4 py-2">Precision / Recall</th>
                  <th className="px-4 py-2">Samples</th>
                  <th className="px-4 py-2">When</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id} className="border-t border-zinc-100 dark:border-zinc-800">
                    <td className="px-4 py-2">{(r.train_accuracy * 100).toFixed(1)}% / {(r.test_accuracy * 100).toFixed(1)}%</td>
                    <td className="px-4 py-2">
                      {r.test_precision != null ? `${(r.test_precision * 100).toFixed(1)}%` : "—"} / {r.test_recall != null ? `${(r.test_recall * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className="px-4 py-2">{r.train_samples} / {r.test_samples}</td>
                    <td className="px-4 py-2 text-xs text-zinc-500">{new Date(r.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function MLStatCard({
  label, accuracy, samples, accent, precision, recall,
}: {
  label: string; accuracy: number; samples: number; accent: "zinc" | "emerald";
  precision?: number | null; recall?: number | null;
}) {
  const border = accent === "emerald" ? "border-emerald-200 dark:border-emerald-900" : "border-zinc-200 dark:border-zinc-800";
  const bg = accent === "emerald" ? "bg-emerald-50 dark:bg-emerald-950" : "bg-zinc-50 dark:bg-zinc-800";
  return (
    <div className={`rounded-lg border ${border} ${bg} p-3`}>
      <p className="text-xs text-zinc-500 dark:text-zinc-400">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{(accuracy * 100).toFixed(1)}%</p>
      <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
        {samples} samples
        {precision != null && recall != null && ` · precision ${(precision * 100).toFixed(1)}% · recall ${(recall * 100).toFixed(1)}%`}
      </p>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-zinc-500">
      {label}
      {children}
    </label>
  );
}
