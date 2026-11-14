/*
  Webcam Rock Paper Scissors — app.js
  1. Collect webcam photos of your hand for each move
  2. Train a small CNN in the browser with TensorFlow.js
  3. Play against an AI that predicts your next move with a Markov model
*/
const NAMES=['Rock','Paper','Scissors'], EMOJI=['✊','✋','✌️'], COLORS=['--rock','--paper','--scissors'];
const S=64, MAX_PER_CLASS=200;
const $=id=>document.getElementById(id);
const video=$('video');
const grab=document.createElement('canvas'); grab.width=grab.height=S;
const gctx=grab.getContext('2d',{willReadFrequently:true});
const samples=[[],[],[]];
let model=null, predictTimer=null, recent=[], busy=false;

/* ---------- camera ---------- */
$('startCam').onclick=async()=>{
  $('camErr').hidden=true;
  try{
    const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:640}},audio:false});
    video.srcObject=stream; await video.play();
    $('camOverlay').style.display='none'; $('frame').hidden=false; $('cam').classList.add('live');
    initTracker();
    $('p1').classList.remove('off');
  }catch(e){
    $('camErr').textContent=(e&&e.name==='NotAllowedError')
      ? 'Camera access was blocked. Allow the camera for this page in your browser settings, then press Start camera again. If it still fails, open this page in its own tab.'
      : 'No camera could be opened ('+(e&&e.name||'unknown error')+'). Check that a webcam is connected and not in use by another app.';
    $('camErr').hidden=false;
  }
};

/* Crop the middle 80% square of the video (matches the dashed frame) into an SxS RGB array */
function snapshot(){
  const vw=video.videoWidth, vh=video.videoHeight; if(!vw) return null;
  const side=Math.min(vw,vh)*0.8, sx=(vw-side)/2, sy=(vh-side)/2;
  gctx.drawImage(video,sx,sy,side,side,0,0,S,S);
  const d=gctx.getImageData(0,0,S,S).data, out=new Float32Array(S*S*3);
  for(let i=0,j=0;i<d.length;i+=4){out[j++]=d[i]/255;out[j++]=d[i+1]/255;out[j++]=d[i+2]/255;}
  return out;
}

/* ---------- step 1: collect ---------- */
let recTimer=null, recClass=-1;
function startRec(c){
  if(recTimer||!video.videoWidth) return;
  recClass=c; $('rec').style.display='block';
  document.querySelector(`.move[data-c="${c}"]`).classList.add('active');
  recTimer=setInterval(()=>{
    if(samples[c].length>=MAX_PER_CLASS) return;
    const s=snapshot(); if(s){samples[c].push(s); updateCounts();}
  },90);
}
function stopRec(){
  if(!recTimer) return; clearInterval(recTimer); recTimer=null;
  $('rec').style.display='none';
  document.querySelectorAll('.move').forEach(b=>b.classList.remove('active'));
}
document.querySelectorAll('.move').forEach(b=>{
  const c=+b.dataset.c;
  b.addEventListener('pointerdown',e=>{e.preventDefault();startRec(c);});
  ['pointerup','pointerleave','pointercancel'].forEach(ev=>b.addEventListener(ev,stopRec));
  b.addEventListener('contextmenu',e=>e.preventDefault());
});
const KEYMAP={r:0,p:1,s:2};
addEventListener('keydown',e=>{if(e.repeat||$('p1').classList.contains('off'))return;const c=KEYMAP[e.key.toLowerCase()];if(c!==undefined)startRec(c);});
addEventListener('keyup',e=>{if(KEYMAP[e.key.toLowerCase()]!==undefined)stopRec();});

function updateCounts(){
  samples.forEach((s,i)=>{$('c'+i).textContent=s.length+(s.length>=MAX_PER_CLASS?' (full)':' samples');const r=$('ring'+i);if(r)r.style.strokeDashoffset=106.8*(1-Math.min(1,s.length/60));document.querySelector(`.move[data-c="${i}"]`).classList.toggle('ready',s.length>=60);});
  const ready=samples.every(s=>s.length>=20);
  $('p1').classList.toggle('done',samples.every(s=>s.length>=60));
  $('trainBtn').disabled=!ready||busy;
  if(!ready) $('trainMsg').textContent='Collect at least 20 samples of each shape to unlock training.';
  else if(!model) $('trainMsg').textContent='A small convolutional network learns from your samples. Takes about 10–30 seconds.';
}
updateCounts();

/* ---------- step 2: train ---------- */
function buildModel(){
  const m=tf.sequential();
  m.add(tf.layers.conv2d({inputShape:[S,S,3],filters:16,kernelSize:3,padding:'same',activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.conv2d({filters:64,kernelSize:3,padding:'same',activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dropout({rate:0.4}));
  m.add(tf.layers.dense({units:64,activation:'relu'}));
  m.add(tf.layers.dense({units:3,activation:'softmax'}));
  m.compile({optimizer:tf.train.adam(0.001),loss:'categoricalCrossentropy',metrics:['accuracy']});
  return m;
}

/* Augment: each sample also appears mirrored with a random brightness shift */
function augmented(src){
  const out=new Float32Array(src.length), k=0.75+Math.random()*0.5;
  for(let y=0;y<S;y++)for(let x=0;x<S;x++){
    const a=(y*S+x)*3, b=(y*S+(S-1-x))*3;
    for(let ch=0;ch<3;ch++) out[b+ch]=Math.min(1,src[a+ch]*k);
  }
  return out;
}

$('trainBtn').onclick=async()=>{
  busy=true; $('trainBtn').disabled=true; stopPredicting();
  $('trainMsg').textContent='Training…'; $('acc').textContent=''; $('prog').style.width='0';
  $('nn').classList.add('on'); accHist.length=0; drawSpark(); $('spark').classList.add('on');
  // Hold out the LAST 15% of each shape's samples for testing (never augmented, never trained on),
  // so the accuracy score reflects photos the model hasn't seen.
  const train=[], val=[];
  samples.forEach((arr,c)=>{
    const cut=Math.floor(arr.length*0.85);
    arr.forEach((s,i)=>{ if(i<cut){train.push([s,c]);train.push([augmented(s),c]);} else val.push([s,c]); });
  });
  for(let i=train.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[train[i],train[j]]=[train[j],train[i]];}
  const toTensors=list=>{
    const flat=new Float32Array(list.length*S*S*3);
    list.forEach(([s],i)=>flat.set(s,i*S*S*3));
    return [tf.tensor4d(flat,[list.length,S,S,3]), tf.oneHot(tf.tensor1d(list.map(t=>t[1]),'int32'),3)];
  };
  const [xs,ys]=toTensors(train), [vx,vy]=toTensors(val);
  if(model) model.dispose();
  model=buildModel();
  const EPOCHS=15;
  try{
    await model.fit(xs,ys,{epochs:EPOCHS,batchSize:32,validationData:[vx,vy],shuffle:true,callbacks:{
      onEpochEnd:async(ep,logs)=>{
        $('prog').style.width=((ep+1)/EPOCHS*100)+'%';
        $('acc').textContent=`Epoch ${ep+1}/${EPOCHS}: ${(logs.val_acc*100).toFixed(0)}% accuracy on photos it never trained on`;
        accHist.push(logs.val_acc); drawSpark();
        await tf.nextFrame();
      }}});
    $('trainMsg').textContent='Trained. If the live read below looks shaky, add more varied samples and train again.';
    $('p3').classList.remove('off'); $('p2').classList.add('done'); $('saveBtn').disabled=false; $('srcCnn').disabled=false; setSource('cnn');
  }catch(e){
    $('trainMsg').textContent='Training failed: '+e.message;
  }finally{
    xs.dispose(); ys.dispose(); vx.dispose(); vy.dispose(); busy=false; updateCounts(); $('nn').classList.remove('on');
  }
};

/* ---------- live prediction ---------- */
$('bars').innerHTML=NAMES.map((n,i)=>`<div class="bar"><span>${EMOJI[i]} ${n}</span><div class="t"><div id="b${i}" style="background:var(${COLORS[i]})"></div></div><span class="v" id="v${i}">0%</span></div>`).join('');
function startPredicting(){
  stopPredicting();
  predictTimer=setInterval(async()=>{
    const s=snapshot(); if(!s) return;
    let probs;
    if(source==='track'){ probs=trackProbs(); if(!probs) return; }
    else{
      if(!model) return;
      const p=tf.tidy(()=>model.predict(tf.tensor4d(s,[1,S,S,3])));
      probs=Array.from(await p.data()); p.dispose();
    }
    recent.push({t:performance.now(),probs,s}); if(recent.length>30) recent.shift();
    probs.forEach((v,i)=>{$('b'+i).style.width=(v*100)+'%';$('v'+i).textContent=Math.round(v*100)+'%';});
    updateHud(probs);
  },100);
}
function stopPredicting(){clearInterval(predictTimer);predictTimer=null;}

/* ---------- AI opponent: learns your patterns with a Markov model ---------- */
const hist=[], table={1:{},2:{},3:{}};
function aiPredict(){
  for(const order of [3,2,1]){
    if(hist.length<order) continue;
    const row=table[order][hist.slice(-order).join('')];
    if(!row) continue;
    const total=row[0]+row[1]+row[2];
    if(total<2) continue;
    const best=row.indexOf(Math.max(...row));
    return {move:best,why:`after your last ${order===1?'throw':order+' throws'}, you went ${NAMES[best].toLowerCase()} ${row[best]} of ${total} times`};
  }
  if(hist.length>=3){
    const f=[0,0,0]; hist.forEach(m=>f[m]++);
    const best=f.indexOf(Math.max(...f));
    return {move:best,why:`${NAMES[best].toLowerCase()} is your most common throw so far`};
  }
  return null;
}
function learn(move){
  hist.push(move);
  for(const order of [1,2,3]){
    if(hist.length<=order) continue;
    const key=hist.slice(-order-1,-1).join('');
    (table[order][key]??=[0,0,0])[move]++;
  }
}


/* After "Shoot!", ignore the first moment while your hand is still moving, then lock in
   the first move the model reads confidently on several frames in a row. */
async function readSteadyHand(){
  const GRACE=300, NEED=3, MIN_CONF=0.7, TIMEOUT=1800;
  const start=performance.now();
  let lastT=start+GRACE, streakMove=-1, streak=0;
  const seen=[];
  while(performance.now()-start<TIMEOUT){
    await sleep(40);
    for(const r of recent.filter(r=>r.t>lastT)){
      lastT=r.t; seen.push(r);
      const c=Math.max(...r.probs), m=r.probs.indexOf(c);
      if(c>=MIN_CONF && m===streakMove) streak++;
      else { streakMove=c>=MIN_CONF?m:-1; streak=streakMove<0?0:1; }
      if(streak>=NEED) return {you:m,conf:c,frame:r.s};
    }
  }
  // Never steady: fall back to the average of the last few frames
  const tail=seen.slice(-4);
  if(!tail.length) return {you:0,conf:0,frame:null};
  const avg=[0,0,0]; tail.forEach(r=>r.probs.forEach((v,i)=>avg[i]+=v/tail.length));
  const conf=Math.max(...avg);
  return {you:avg.indexOf(conf),conf,frame:tail[tail.length-1].s};
}

/* Draw the exact 64x64 photo the model judged, so you can check its call */
function showSeen(frame,label){
  if(!frame){ $('seen').hidden=true; return; }
  const ctx=$('seenCanvas').getContext('2d'), img=ctx.createImageData(S,S);
  for(let i=0,j=0;j<frame.length;i+=4){
    img.data[i]=frame[j++]*255; img.data[i+1]=frame[j++]*255; img.data[i+2]=frame[j++]*255; img.data[i+3]=255;
  }
  ctx.putImageData(img,0,0);
  $('seenText').textContent=label; $('seen').hidden=false;
}

/* ---------- game loop ---------- */
const score={w:0,d:0,l:0};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
$('playBtn').onclick=async()=>{
  if(busy) return; busy=true; $('playBtn').disabled=true; $('trainBtn').disabled=true;
  $('youHand').textContent=''; $('aiHand').textContent=''; $('verdict').textContent=''; $('why').textContent=''; $('seen').hidden=true;
  // AI commits before seeing your hand
  const guess=Math.random()<0.1?null:aiPredict();
  const aiMove=guess?(guess.move+1)%3:Math.floor(Math.random()*3);
  $('arena').className='arena'; $('youHand').textContent='✊'; $('aiHand').textContent='✊'; $('youHand').className='hand bob'; $('aiHand').className='hand bob';
  for(const w of ['Rock','Paper','Scissors']){showCount(w,'go');await sleep(550);}
  showCount('Shoot!','shoot'); $('flash').classList.remove('on'); void $('flash').offsetWidth; $('flash').classList.add('on'); $('cam').classList.remove('shake'); void $('cam').offsetWidth; $('cam').classList.add('shake');
  const {you,conf,frame}=await readSteadyHand();
  $('count').textContent=''; $('youHand').className='hand'; $('aiHand').className='hand'; $('youHand').textContent=''; $('aiHand').textContent='';
  showSeen(frame, conf<0.5 ? 'What the camera saw at “Shoot!”' : `What the model saw at “Shoot!”: ${NAMES[you]}, ${Math.round(conf*100)}% sure`);
  if(conf<0.5){
    $('verdict').textContent='Couldn’t read your hand';
    $('why').textContent='Your hand never held still long enough. Throw on “Shoot!” and hold the shape inside the square for a moment.';
  }else{
    $('youHand').textContent=EMOJI[you]; $('aiHand').textContent=EMOJI[aiMove]; $('youHand').className='hand reveal'; $('aiHand').className='hand reveal';
    const r=(you-aiMove+3)%3;
    const res=r===0?'d':r===1?'w':'l';
    score[res]++;
    $('verdict').textContent=res==='w'?'You win':res==='l'?'AI wins':'Draw';
    $('verdict').style.color=res==='w'?'var(--scissors)':res==='l'?'var(--lose)':'var(--ink)';
    $('verdict').classList.remove('in'); void $('verdict').offsetWidth; $('verdict').classList.add('in');
    $('arena').className='arena '+res; streak=res==='w'?streak+1:0;
    $('streak').textContent=streak>=2?`🔥 ${streak} wins in a row`:''; if(streak>=2){$('streak').classList.remove('in');void $('streak').offsetWidth;$('streak').classList.add('in');}
    if(res==='w') confetti(streak>=3?220:120);
    $('why').textContent=guess
      ? `The AI expected ${NAMES[guess.move].toLowerCase()}: ${guess.why}.`
      : 'The AI threw at random this round.';
    const h=document.createElement('i'); h.className=res; h.textContent=EMOJI[you]; h.title=`You: ${NAMES[you]}, AI: ${NAMES[aiMove]}`;
    $('history').appendChild(h);
    learn(you);
    $('sW').textContent=score.w; $('sD').textContent=score.d; $('sL').textContent=score.l;
    const bid={w:'sW',d:'sD',l:'sL'}[res]; $(bid).classList.remove('bump'); void $(bid).offsetWidth; $(bid).classList.add('bump');
  }
  busy=false; $('playBtn').disabled=false; updateCounts();
};
$('resetBtn').onclick=()=>{
  score.w=score.d=score.l=0; hist.length=0; table[1]={};table[2]={};table[3]={};
  ['sW','sD','sL'].forEach(id=>$(id).textContent='0'); $('history').innerHTML=''; streak=0; $('streak').textContent=''; $('arena').className='arena';
  $('youHand').textContent=''; $('aiHand').textContent=''; $('verdict').textContent=''; $('why').textContent='';
};

/* ---------- save / load the trained model (browser storage) ---------- */
const MODEL_KEY='localstorage://rps-webcam-model';
$('saveBtn').onclick=async()=>{
  if(!model) return;
  try{ await model.save(MODEL_KEY); $('saveMsg').textContent='Saved in this browser.'; }
  catch(e){ $('saveMsg').textContent='Could not save: '+e.message; }
};
$('loadBtn').onclick=async()=>{
  try{
    const m=await tf.loadLayersModel(MODEL_KEY);
    if(model) model.dispose();
    model=m; $('saveBtn').disabled=false; $('p2').classList.add('done'); $('srcCnn').disabled=false; setSource('cnn');
    $('saveMsg').textContent='Loaded your saved model.';
    $('p3').classList.remove('off');
    if(video.videoWidth) startPredicting();
    else $('saveMsg').textContent='Loaded. Start the camera to play.';
  }catch(e){ $('saveMsg').textContent='No saved model found yet. Train one and press Save model.'; }
};
// If the model was loaded before the camera started, begin predicting once video is live
video.addEventListener('playing',()=>{ if(model||source==='track') startPredicting(); });

/* ---------- visual effects ---------- */
let streak=0, lastTop=-1;
const accHist=[];
function showCount(text,cls){const c=$('count');c.textContent=text;c.className='count';void c.offsetWidth;c.className='count '+cls;}
function drawSpark(){
  const n=accHist.length; if(!n){$('sparkLine').setAttribute('points','');return;}
  const pts=accHist.map((v,i)=>`${n<2?0:i/(n-1)*300},${48-v*44}`).join(' ');
  $('sparkLine').setAttribute('points',n<2?pts+' 300,'+(48-accHist[0]*44):pts);
}
const HEX={0:'#8F86FF',1:'#FFC24B',2:'#3FE0C0'};
function updateHud(probs){
  const c=Math.max(...probs), m=probs.indexOf(c), hud=$('hud'), fr=$('frame');
  hud.hidden=false;
  document.querySelectorAll('.bar').forEach((b,i)=>b.classList.toggle('top',i===m&&c>=.5));
  if(c>=.7){
    $('hudE').textContent=EMOJI[m]; $('hudT').textContent=`${NAMES[m]} · ${Math.round(c*100)}%`;
    fr.style.setProperty('--fc',HEX[m]); fr.classList.add('lock');
    if(m!==lastTop){hud.classList.remove('pop');void hud.offsetWidth;hud.classList.add('pop');lastTop=m;}
  }else{
    $('hudE').textContent='🤔'; $('hudT').textContent='Not sure yet…';
    fr.style.setProperty('--fc','rgba(255,255,255,.8)'); fr.classList.remove('lock'); lastTop=-1;
  }
}
/* neural network diagram for the training panel */
(function drawNN(){
  const svg=$('nnSvg'), layers=[[3,'input'],[5,'conv'],[5,'conv'],[4,'dense'],[3,'output']], ns='http://www.w3.org/2000/svg';
  const pos=layers.map(([k],li)=>Array.from({length:k},(_,j)=>({x:30+li*85,y:12+(j+.5)*(92/k)})));
  pos.forEach((L,li)=>{if(li<pos.length-1)L.forEach(a=>pos[li+1].forEach(b=>{const l=document.createElementNS(ns,'line');l.setAttribute('x1',a.x);l.setAttribute('y1',a.y);l.setAttribute('x2',b.x);l.setAttribute('y2',b.y);l.style.animationDelay=(Math.random()*-1.4)+'s';svg.appendChild(l);}));});
  pos.forEach((L,li)=>{L.forEach((p,j)=>{const c=document.createElementNS(ns,'circle');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',li===4?7:5.5);c.style.animationDelay=(li*.18+j*.05)+'s';svg.appendChild(c);});
    const t=document.createElementNS(ns,'text');t.setAttribute('x',pos[li][0].x);t.setAttribute('y',116);t.textContent=layers[li][1];svg.appendChild(t);});
})();
/* confetti */
const cv=$('confetti'), cx=cv.getContext('2d'); let parts=[], raf=0;
function confetti(n){
  if(matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  cv.width=innerWidth*devicePixelRatio; cv.height=innerHeight*devicePixelRatio;
  const cols=['#8F86FF','#FFC24B','#3FE0C0','#FF5C7A','#ffffff'];
  for(let i=0;i<n;i++) parts.push({x:cv.width/2+(Math.random()-.5)*cv.width*.3,y:cv.height*.35,vx:(Math.random()-.5)*22,vy:-Math.random()*20-8,r:Math.random()*Math.PI,vr:(Math.random()-.5)*.4,s:(6+Math.random()*8)*devicePixelRatio,c:cols[i%cols.length],life:0});
  if(!raf) raf=requestAnimationFrame(tick);
}
function tick(){
  cx.clearRect(0,0,cv.width,cv.height);
  parts.forEach(p=>{p.vy+=.6;p.vx*=.99;p.x+=p.vx;p.y+=p.vy;p.r+=p.vr;p.life++;
    cx.save();cx.translate(p.x,p.y);cx.rotate(p.r);cx.fillStyle=p.c;cx.fillRect(-p.s/2,-p.s/4,p.s,p.s/2);cx.restore();});
  parts=parts.filter(p=>p.y<cv.height+40&&p.life<240);
  raf=parts.length?requestAnimationFrame(tick):0;
}


/* ---------- hand tracking (MediaPipe): reads rock / paper / scissors from finger joints ---------- */
let source='track', tracker=null, trackRes=null, trackLoop=0, lastVT=-1;
const MP='https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
async function initTracker(){
  $('srcMsg').textContent='Loading hand tracking…';
  try{
    const {FilesetResolver,HandLandmarker}=await import(MP+'/vision_bundle.mjs');
    const fs=await FilesetResolver.forVisionTasks(MP+'/wasm');
    const opts=d=>({baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',delegate:d},runningMode:'VIDEO',numHands:1,minHandDetectionConfidence:.5,minHandPresenceConfidence:.5,minTrackingConfidence:.5});
    try{ tracker=await HandLandmarker.createFromOptions(fs,opts('GPU')); }
    catch(e){ tracker=await HandLandmarker.createFromOptions(fs,opts('CPU')); }
    $('p3').classList.remove('off');
    $('srcMsg').textContent='Hand tracking is on: it finds the 21 joints of your hand and checks which fingers are straight. No training needed.';
    trackFrame();
    if(source==='track') startPredicting();
  }catch(e){
    console.error(e); tracker=null;
    $('srcTrack').disabled=true; source='cnn'; $('srcTrack').classList.remove('on');
    $('srcMsg').textContent='Hand tracking could not load on this device, so train your own model in steps 1 and 2.';
  }
}
function trackFrame(){
  if(source==='track'&&tracker&&video.readyState>=2&&video.currentTime!==lastVT){
    lastVT=video.currentTime;
    try{ const r=tracker.detectForVideo(video,performance.now()); trackRes=r.landmarks&&r.landmarks[0]?r.landmarks[0]:null; }catch(e){ trackRes=null; }
    drawSkeleton();
  }
  trackLoop=requestAnimationFrame(trackFrame);
}
const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y,(a.z-b.z)*.5);
const ext=(L,tip,pip,mcp)=>{ // 0 = curled, 1 = straight
  const r=dist(L[0],L[tip])/Math.max(1e-6,dist(L[0],L[pip]));
  const s=Math.min(1,Math.max(0,(r-1.0)/0.3));
  const r2=dist(L[mcp],L[tip])/Math.max(1e-6,dist(L[mcp],L[pip])); // tip beyond the middle joint
  return Math.min(s,Math.min(1,Math.max(0,(r2-1.2)/0.5)));
};
function trackProbs(){
  if(!trackRes) return null;
  const L=trackRes, i=ext(L,8,6,5), m=ext(L,12,10,9), r=ext(L,16,14,13), p=ext(L,20,18,17);
  const paper=(i*m*r*p)**.5*.6+((i+m+r+p)/4)**2*.4;
  const scissors=i*m*(1-r)*(1-p);
  const rock=(1-i)*(1-m)*(1-r)*(1-p)*.7+((4-i-m-r-p)/4)**2*.3;
  const sum=rock+paper+scissors+1e-6;
  return [rock/sum,paper/sum,scissors/sum];
}
const BONES=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
function drawSkeleton(){
  const c=$('skel'), ctx=c.getContext('2d'), W=c.clientWidth*devicePixelRatio, H=c.clientHeight*devicePixelRatio;
  if(c.width!==W||c.height!==H){c.width=W;c.height=H;}
  ctx.clearRect(0,0,W,H); if(!trackRes) return;
  // video is object-fit:cover in a square; map normalized video coords to the square
  const vw=video.videoWidth, vh=video.videoHeight, sc=Math.max(W/vw,H/vh), ox=(W-vw*sc)/2, oy=(H-vh*sc)/2;
  const P=L=>[ox+L.x*vw*sc, oy+L.y*vh*sc];
  const col=getComputedStyle($('frame')).getPropertyValue('--fc')||'#8F86FF';
  ctx.lineCap='round'; ctx.shadowColor=col; ctx.shadowBlur=14*devicePixelRatio;
  ctx.strokeStyle=col; ctx.lineWidth=4*devicePixelRatio; ctx.globalAlpha=.9;
  ctx.beginPath(); for(const [a,b] of BONES){const [x1,y1]=P(trackRes[a]),[x2,y2]=P(trackRes[b]);ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);} ctx.stroke();
  ctx.shadowBlur=0; ctx.globalAlpha=1;
  trackRes.forEach((L,k)=>{const [x,y]=P(L);ctx.fillStyle=[4,8,12,16,20].includes(k)?'#fff':col;ctx.beginPath();ctx.arc(x,y,(k%4===0?5.5:3.5)*devicePixelRatio,0,7);ctx.fill();});
}
function setSource(s){
  source=s; $('srcTrack').classList.toggle('on',s==='track'); $('srcCnn').classList.toggle('on',s==='cnn');
  $('srcMsg').textContent=s==='track'?'Hand tracking is on: it finds the 21 joints of your hand and checks which fingers are straight. No training needed.':'Using the CNN you trained on your own photos.';
  if(s==='cnn'){ trackRes=null; const c=$('skel'); c.getContext('2d').clearRect(0,0,c.width,c.height); }
  startPredicting();
}
$('srcTrack').onclick=()=>{ if(tracker) setSource('track'); };
$('srcCnn').onclick=()=>{ if(model) setSource('cnn'); };
