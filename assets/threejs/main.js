const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function recordAnimation(folderPath) {
    const downloadPath = path.join(folderPath, 'downloads');

    const browser = await puppeteer.launch({
        headless: true,
        defaultViewport: null,
        args: [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-gpu",
            "--enable-unsafe-swiftshader",
            "--allow-file-access-from-files"
        ]
    });

    try {
        const page = await browser.newPage();
        let recordingComplete = false;
        
        // 立即设置事件监听器
        page.on("pageerror", err => {
            console.log(`[PAGE ERROR] ${err.message}`);
            console.log(`[PAGE ERROR] ${err.stack}`);
        });
        
        page.on('console', msg => {
            console.log(`[${msg.type().toUpperCase()}] ${msg.text()}`);
            const text = msg.text() || '';
            if (text.includes('Video download initiated') || text.includes('Recording stopped')) {
                recordingComplete = true;
            }
        });
        
        page.on('error', err => {
            console.log(`[ERROR] ${err.message}`);
        });
        
        page.on('load', () => {
            console.log(`[LOAD] Page loaded successfully`);
        });

        const client = await page.createCDPSession();
        await client.send('Page.setDownloadBehavior', {
            behavior: 'allow',
            downloadPath: downloadPath
        });

        const htmlPath = path.join(folderPath, 'index.html');
        console.log(`[NAVIGATE] Loading: file://${path.resolve(htmlPath)}`);
        
        await page.goto(`file://${path.resolve(htmlPath)}`, { 
            waitUntil: 'domcontentloaded',
            timeout: 30000
        });
        
        console.log(`[NAVIGATE] Page navigation completed`);
        
        // 等待一点时间让 JavaScript 执行
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // 智能等待录制完成 - 检查控制台输出 + 轮询文件落盘
        let waitTime = 0;
        const maxWaitTime = 120000; // 最多等待120秒（支持更长录制）
        const fsPollInterval = 250;

        // 轮询等待录制完成
        while (!recordingComplete && waitTime < maxWaitTime) {
            await new Promise(resolve => setTimeout(resolve, fsPollInterval));
            waitTime += fsPollInterval;
            try {
                const files = fs.readdirSync(downloadPath);
                if (files.some(f => f.endsWith('.webm'))) {
                    recordingComplete = true;
                }
            } catch (e) {
                // ignore
            }
        }
        
        if (!recordingComplete) {
            console.log('⚠️ 录制状态检测超时，使用固定等待');
        }

        // 额外缓冲时间确保文件写入完成
        await new Promise(resolve => setTimeout(resolve, 2000));

        const files = fs.readdirSync(downloadPath);
        const videoFile = files.find(file => file.endsWith('.webm'));

        if (videoFile) {
            const oldPath = path.join(downloadPath, videoFile);
            const newPath = path.join(downloadPath, "output.webm");
            
            // 只有在文件名不是 output.webm 时才重命名
            if (videoFile !== 'output.webm') {
                fs.renameSync(oldPath, newPath);
                console.log(`Video recorded and renamed from ${videoFile} to: output.webm`);
            } else {
                console.log(`Video recorded as: output.webm`);
            }
            console.log(`Saved to: ${newPath}`);
        } else {
            console.error('Video could not be recorded! Available files:', files);
            process.exit(1);
        }

    } catch (error) {
        console.error('An error occurred:', error);
    } finally {
        await browser.close();
    }
}

const folderPath = process.argv[2];

if (!folderPath) {
    console.error('Please provide the folder path as a command line argument');
    process.exit(2);
}

recordAnimation(folderPath).catch(console.error);
