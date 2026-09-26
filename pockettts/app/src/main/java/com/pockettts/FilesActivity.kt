package com.pockettts

import android.app.Activity
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView

/**
 * File management screen: every file the synthesizer needs, grouped by type
 * (Models / Voices / Host assets), with a category filter, a per-category
 * Download-missing and Delete, plus a global Delete-all. Downloads go to the
 * app's files dir and persist, so a subsequent launch needs no network.
 */
class FilesActivity : Activity() {

    private var store: ModelStore? = null

    // Filter positions match 0=All, then ModelStore.Category order.
    private val filters = listOf<ModelStore.Category?>(null) +
        ModelStore.Category.entries.toList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = ModelStore(this)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 48, 32, 32)
        }
        val title = TextView(this).apply {
            text = "Files"
            textSize = 22f
        }
        val subtitle = TextView(this).apply {
            text = "Models, voices and host assets for the synth. Downloaded once, stored on this phone."
            textSize = 13f
            alpha = 0.7f
        }
        val filter = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@FilesActivity,
                android.R.layout.simple_spinner_dropdown_item,
                filters.map { it?.label ?: "All files" },
            )
        }
        val list = TextView(this).apply { textSize = 13f }
        val download = Button(this).apply { text = "Download missing" }
        val deleteShown = Button(this).apply { text = "Delete shown" }
        val deleteAll = Button(this).apply { text = "Delete all files" }
        root.addView(title)
        root.addView(subtitle, LinearLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(filter, LinearLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = 24 })
        root.addView(list, LinearLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = 24 })
        root.addView(download, LinearLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = 24 })
        root.addView(deleteShown, LinearLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = 12 })
        root.addView(deleteAll, LinearLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = 12 })
        setContentView(ScrollView(this).apply { addView(root) })

        fun refresh() {
            val s = store ?: return
            val cat = filters[filter.selectedItemPosition]
            val rows = if (cat == null) s.REQUIRED
                       else s.REQUIRED.filter { it.category == cat }
            val sb = StringBuilder()
            sb.append("${if (cat == null) "All categories" else cat.label}\n\n")
            for (m in rows) {
                val present = s.isPresent(m.name)
                val size = s.localSize(m.name)
                val sz = if (size < 0) "" else " (${size / 1048576} MiB)"
                val sym = when {
                    present -> "●"
                    m.bundled -> "◆"      // bundled = copied from APK on first use
                    m.onHf -> "○"
                    else -> "—"
                }
                sb.append("$sym ${m.name}$sz\n    ${m.note}\n")
            }
            val missing = rows.count { it.hfPath.isNotEmpty() && !s.isPresent(it.name) }
            val present = rows.count { s.isPresent(it.name) }
            val bundled = rows.count { it.bundled }
            sb.append("\n$present present; $bundled bundled; $missing downloadable-missing" +
                if (cat == null) " ${s.statusSummary()}" else "")
            val dlEnabled = missing > 0
            download.isEnabled = dlEnabled
            deleteShown.isEnabled = rows.any { s.isPresent(it.name) }
            deleteAll.isEnabled = true
            list.text = sb.toString()
        }

        filter.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: android.widget.AdapterView<*>?, v: android.view.View?,
                pos: Int, id: Long,
            ) { refresh() }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }

        download.setOnClickListener {
            val s = store ?: return@setOnClickListener
            val cat = filters[filter.selectedItemPosition]
            download.isEnabled = false
            deleteShown.isEnabled = false
            deleteAll.isEnabled = false
            list.text = "Downloading…"
            // downloadMissing is all-hosts anyway (per-category subset is
            // cosmetic; the pool stays the same). Filter the callback result.
            s.downloadMissing(
                onProgress = { done, total, _ -> list.text = "Downloading… $done/$total" },
            ) { dl, fail ->
                runOnUiThread {
                    val shown = when (cat) {
                        null -> s.REQUIRED
                        else -> s.REQUIRED.filter { it.category == cat }
                    }
                    val dlIt = shown.filter { it.name in dl }
                    list.text = "Downloaded: ${dlIt.size}" +
                        (if (fail.isNotEmpty()) ", FAILED: ${fail.joinToString()} " else "") +
                        if (fail.isEmpty()) "\n" else ""
                    download.isEnabled = true
                    deleteShown.isEnabled = true
                    deleteAll.isEnabled = true
                    refresh()
                }
            }
        }

        deleteShown.setOnClickListener {
            val s = store ?: return@setOnClickListener
            val cat = filters[filter.selectedItemPosition]
            if (cat == null) s.deleteAll() else s.deleteCategory(cat)
            refresh()
        }

        deleteAll.setOnClickListener {
            store?.deleteAll()
            refresh()
        }

        refresh()
    }
}