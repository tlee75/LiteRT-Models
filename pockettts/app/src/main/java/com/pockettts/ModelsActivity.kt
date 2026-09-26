package com.pockettts

import android.app.Activity
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * Model management screen: lists every file the synthesizer needs, shows
 * present/missing status, and offers one Download-missing button and one
 * Delete-all button. Downloads go to the app's files dir and persist, so a
 * subsequent launch needs no network.
 */
class ModelsActivity : Activity() {

    private var store: ModelStore? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = ModelStore(this)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 48, 32, 32)
        }
        val title = TextView(this).apply {
            text = "Models"
            textSize = 22f
        }
        val subtitle = TextView(this).apply {
            text = "Downloaded once, stored on this phone; delete to re-fetch."
            textSize = 13f
            alpha = 0.7f
        }
        val list = TextView(this).apply {
            textSize = 13f
        }
        val download = Button(this).apply { text = "Download missing" }
        val deleteAll = Button(this).apply { text = "Delete all models" }
        root.addView(title)
        root.addView(subtitle, LinearLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(list, LinearLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = 32 })
        root.addView(download, LinearLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = 24 })
        root.addView(deleteAll, LinearLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = 12 })
        setContentView(ScrollView(this).apply { addView(root) })

        fun refresh() {
            val s = store ?: return
            val sb = StringBuilder()
            for (m in s.REQUIRED) {
                val present = s.isPresent(m.name)
                val size = s.localSize(m.name)
                val sz = if (size < 0) "" else " (${size / 1048576} MiB)"
                val sym = if (present) "●" else if (m.onHf) "○" else "—"
                sb.append("$sym ${m.name}$sz\n    ${m.note}\n")
            }
            sb.append("\n${s.statusSummary()}")
            list.text = sb.toString()
        }

        download.setOnClickListener {
            val s = store ?: return@setOnClickListener
            download.isEnabled = false
            deleteAll.isEnabled = false
            list.text = "Downloading…"
            s.downloadMissing(
                onProgress = { done, total, _ -> list.text = "Downloading… $done/$total" },
            ) { dl, fail ->
                runOnUiThread {
                    list.text = "Downloaded: ${dl.size}" +
                        (if (fail.isNotEmpty()) ", FAILED: ${fail.joinToString()} " else "") +
                        if (fail.isEmpty()) "\n\n"+ buildString {
                            for (m in s.REQUIRED) {
                                if (s.isPresent(m.name))
                                    append("● ${m.name} (${s.localSize(m.name) / 1048576} MiB)\n")
                            }
                        } else ""
                    download.isEnabled = true
                    deleteAll.isEnabled = true
                    refresh()
                }
            }
        }

        deleteAll.setOnClickListener {
            store?.deleteAll()
            refresh()
        }

        refresh()
    }
}
