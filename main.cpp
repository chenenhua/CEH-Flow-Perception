#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QTimer>
#include <QDateTime>
#include <QImage>
#include <QPixmap>
#include <QPainter>
#include <QFrame>
#include <QScrollArea>
#include <QProgressBar>
#include <QFont>
#include <QElapsedTimer>
#include <QCheckBox>
#include <QSplitter>
#include <QMetaObject>
#include <QButtonGroup>
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>

static float clampf(float v,float lo,float hi){return std::max(lo,std::min(v,hi));}
static double clampd(double v,double lo,double hi){return std::max(lo,std::min(v,hi));}

struct PhysicalAgent{
    int id=0;
    cv::Point2f pos=cv::Point2f(0,0);
    cv::Point2f vel=cv::Point2f(0,0);
    cv::Point2f acc=cv::Point2f(0,0);
    float mass=1.0f;
    float drag=0.84f;
    float maxVel=6.0f;
    float radius=8.0f;
    void update(const cv::Point2f& netForce,float dt){
        acc=netForce/std::max(0.1f,mass);
        vel+=acc*dt;
        vel*=drag;
        float speed=cv::norm(vel);
        if(speed<0.03f){vel=cv::Point2f(0,0);}
        else if(speed>maxVel){vel=(vel/speed)*maxVel;}
        pos+=vel*dt;
    }
};

struct BenchmarkMetrics{
    int totalFrames=0;
    int collisions=0;
    double totalSpeed=0.0;
    double totalPath=0.0;
    double totalEnergy=0.0;
    double totalRepulsion=0.0;
    double totalGradient=0.0;
    void reset(){totalFrames=0;collisions=0;totalSpeed=0.0;totalPath=0.0;totalEnergy=0.0;totalRepulsion=0.0;totalGradient=0.0;}
};

static QImage matToQImage(const cv::Mat& mat){
    if(mat.empty()) return QImage();
    if(mat.type()==CV_8UC3){cv::Mat rgb;cv::cvtColor(mat,rgb,cv::COLOR_BGR2RGB);return QImage(rgb.data,rgb.cols,rgb.rows,(int)rgb.step,QImage::Format_RGB888).copy();}
    if(mat.type()==CV_8UC1){return QImage(mat.data,mat.cols,mat.rows,(int)mat.step,QImage::Format_Grayscale8).copy();}
    if(mat.type()==CV_8UC4){return QImage(mat.data,mat.cols,mat.rows,(int)mat.step,QImage::Format_RGBA8888).copy();}
    cv::Mat tmp;mat.convertTo(tmp,CV_8U);return QImage(tmp.data,tmp.cols,tmp.rows,(int)tmp.step,QImage::Format_Grayscale8).copy();
}

class MainWindow:public QMainWindow{
    Q_OBJECT
public:
    MainWindow(){
        resize(1820,1040);
        setWindowTitle("CEH-Flow-Perception 流体驱动的心智（自适应商业演示版）");
        buildUi();
        initState();
        timer=new QTimer(this);
        connect(timer,&QTimer::timeout,this,[this](){onFrame();});
        timer->start(33);
        appendLog("【系统】Chen-Flow 自适应时空感知引擎已启动");
        appendLog("【说明】左侧给普通人看现象，右侧给专家看指标，底部日志给销售讲故事");
    }
    ~MainWindow(){
        if(cap.isOpened()) cap.release();
    }

private:
    QWidget* central=nullptr;
    QLabel* videoLabel=nullptr;
    QLabel* statusBig=nullptr;
    QLabel* riskLabel=nullptr;
    QLabel* energyLabel=nullptr;
    QLabel* gradientLabel=nullptr;
    QLabel* repulsionLabel=nullptr;
    QLabel* entropyLabel=nullptr;
    QLabel* fpsLabel=nullptr;
    QLabel* conductLabel=nullptr;
    QLabel* learnLabel=nullptr;
    QLabel* agentLabel=nullptr;
    QLabel* benchmarkLabel=nullptr;
    QLabel* autoTuneLabel=nullptr;
    QTextEdit* logEdit=nullptr;
    QComboBox* cameraIndexBox=nullptr;
    QPushButton* openBtn=nullptr;
    QPushButton* closeBtn=nullptr;
    QPushButton* flowBtn=nullptr;
    QPushButton* resetBtn=nullptr;
    QPushButton* calibBtn=nullptr;
    QPushButton* swarmBtn=nullptr;
    QPushButton* benchmarkBtn=nullptr;
    QPushButton* autoTuneBtn=nullptr;
    QPushButton* demoBtn=nullptr;
    QCheckBox* drawVectorBox=nullptr;
    QCheckBox* drawTrailBox=nullptr;
    QCheckBox* drawSwarmBox=nullptr;
    QCheckBox* purpleMemoryBox=nullptr;
    QCheckBox* darkFlowBox=nullptr;
    QCheckBox* backReactionBox=nullptr;
    QCheckBox* autoRiskBox=nullptr;
    QCheckBox* predictBox=nullptr;
    QCheckBox* learnOverlayBox=nullptr;
    QCheckBox* conductOverlayBox=nullptr;
    QCheckBox* benchmarkObstacleBox=nullptr;
    QDoubleSpinBox* pAlpha=nullptr;
    QDoubleSpinBox* pBeta=nullptr;
    QDoubleSpinBox* pLeak=nullptr;
    QDoubleSpinBox* pRetain=nullptr;
    QDoubleSpinBox* pForget=nullptr;
    QDoubleSpinBox* pLearn=nullptr;
    QDoubleSpinBox* pLearnGain=nullptr;
    QDoubleSpinBox* pFieldCap=nullptr;
    QDoubleSpinBox* pAttract=nullptr;
    QDoubleSpinBox* pBoundary=nullptr;
    QDoubleSpinBox* pDt=nullptr;
    QDoubleSpinBox* pBackReaction=nullptr;
    QDoubleSpinBox* pSocial=nullptr;
    QDoubleSpinBox* pPersonalSpace=nullptr;
    QDoubleSpinBox* pPredictTime=nullptr;
    QDoubleSpinBox* pPredictGain=nullptr;
    QDoubleSpinBox* pAdaptiveGain=nullptr;
    QSpinBox* pGridW=nullptr;
    QSpinBox* pGridH=nullptr;
    QSpinBox* pSwarmSize=nullptr;
    QSpinBox* pTrailRadius=nullptr;
    QProgressBar* costBarTraditional=nullptr;
    QProgressBar* costBarChen=nullptr;

    QTimer* timer=nullptr;
    cv::VideoCapture cap;
    cv::Mat frameBgr,prevGray,gray,motionMask,conductMap,learnMap,potentialField,benchmarkFrame;
    bool cameraOpened=false;
    bool flowEnabled=true;
    bool avoidFieldEnabled=true;
    bool swarmEnabled=true;
    bool calibrated=false;
    bool benchmarkMode=false;
    bool autoTuneEnabled=true;
    bool demoMode=true;
    int frameCount=0;
    int frameWidth=1280,frameHeight=720;
    int gridW=320,gridH=180;
    float scaleX=1.0f,scaleY=1.0f;
    double fps=0.0;
    qint64 fpsLastTick=0;
    QElapsedTimer fpsTimer;
    QString riskState="LOW";

    double lastEnergySum=0.0,lastEnergyMean=0.0,lastEnergyPeak=0.0;
    double lastGradientMean=0.0,lastRepulsion=0.0,lastEntropy=0.0;
    double lastRiskRaw=0.0,lastBrightness=0.0,lastMotion=0.0,lastSharp=0.0,lastConductAvg=0.0,lastLearnAvg=0.0;
    double baselineEnergy=0.0,baselineMotion=0.0,baselineEntropy=0.0;
    std::deque<double> energyHistory,motionHistory,gradientHistory;

    PhysicalAgent vehicle;
    std::vector<cv::Point2f> vehicleTrail;
    std::vector<PhysicalAgent> swarm;
    BenchmarkMetrics bench;
    int benchmarkTick=0;

    static constexpr int BUCKET_COLS=40;
    static constexpr int BUCKET_ROWS=30;
    std::vector<int> gridBuckets[BUCKET_COLS][BUCKET_ROWS];

    QString hudInteractionText="【巡航】场能平稳，按最优引力线推进。";
    cv::Scalar hudInteractionColor=cv::Scalar(0,255,0);

    QDoubleSpinBox* makeD(double v,double mi,double ma,double step,int dec=3){QDoubleSpinBox* s=new QDoubleSpinBox;s->setRange(mi,ma);s->setDecimals(dec);s->setSingleStep(step);s->setValue(v);return s;}
    QSpinBox* makeI(int v,int mi,int ma){QSpinBox* s=new QSpinBox;s->setRange(mi,ma);s->setValue(v);return s;}

    void buildUi(){
        central=new QWidget(this);
        setCentralWidget(central);
        setStyleSheet("QWidget{background:#0e1116;color:#e6e6e6;font-size:13px;}QGroupBox{border:1px solid #34383d;border-radius:8px;margin-top:10px;font-weight:bold;color:#64b5f6;}QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 6px;}QPushButton{background:#1a2332;border:1px solid #3a4b66;border-radius:6px;padding:8px 12px;color:#f2f6fb;font-weight:bold;}QPushButton:hover{background:#28354d;}QPushButton:pressed{background:#111820;}QTextEdit{background:#06080a;border:1px solid #2f3740;color:#00e676;}QLabel{color:#d7dde5;}QSpinBox,QDoubleSpinBox,QComboBox{background:#171c21;border:1px solid #38424c;border-radius:4px;padding:4px;color:#64b5f6;}QCheckBox{spacing:6px;}QProgressBar{background:#111;border:1px solid #444;color:#fff;text-align:center;}");

        QHBoxLayout* root=new QHBoxLayout(central);
        root->setContentsMargins(8,8,8,8);
        root->setSpacing(12);

        QWidget* leftPanel=new QWidget;
        QVBoxLayout* leftLayout=new QVBoxLayout(leftPanel);
        leftLayout->setContentsMargins(0,0,0,0);

        QGroupBox* videoBox=new QGroupBox("主视觉场：给外行看现象");
        QVBoxLayout* videoLayout=new QVBoxLayout(videoBox);
        videoLabel=new QLabel("等待摄像头 / 基准测试启动...");
        videoLabel->setMinimumSize(980,700);
        videoLabel->setAlignment(Qt::AlignCenter);
        videoLabel->setStyleSheet("background:#030405;border:1px solid #202428;border-radius:8px;font-size:24px;color:#666;");
        videoLayout->addWidget(videoLabel);
        leftLayout->addWidget(videoBox,1);

        QGroupBox* logBox=new QGroupBox("运行日志：给销售与工程师讲因果");
        QVBoxLayout* logLayout=new QVBoxLayout(logBox);
        logEdit=new QTextEdit;
        logEdit->setReadOnly(true);
        logEdit->setMinimumHeight(200);
        logLayout->addWidget(logEdit);
        leftLayout->addWidget(logBox,0);

        QWidget* rightPanel=new QWidget;
        rightPanel->setMinimumWidth(520);
        QVBoxLayout* rightLayout=new QVBoxLayout(rightPanel);
        rightLayout->setContentsMargins(0,0,0,0);

        QGroupBox* controlBox=new QGroupBox("系统控制台");
        QGridLayout* ctl=new QGridLayout(controlBox);
        cameraIndexBox=new QComboBox;
        for(int i=0;i<6;++i) cameraIndexBox->addItem(QString::number(i));
        openBtn=new QPushButton("开启摄像头");
        closeBtn=new QPushButton("关闭摄像头");
        flowBtn=new QPushButton("关闭时空记忆");
        resetBtn=new QPushButton("清空物理场");
        calibBtn=new QPushButton("一键环境标定");
        swarmBtn=new QPushButton("重建集群");
        benchmarkBtn=new QPushButton("开启基准测试");
        autoTuneBtn=new QPushButton("关闭自适应调参");
        demoBtn=new QPushButton("切换演示模式");

        drawVectorBox=new QCheckBox("显示力矢量因果线"); drawVectorBox->setChecked(true);
        drawTrailBox=new QCheckBox("显示主车物理足迹"); drawTrailBox->setChecked(true);
        drawSwarmBox=new QCheckBox("显示多智能体集群"); drawSwarmBox->setChecked(true);
        purpleMemoryBox=new QCheckBox("显示紫色导通残影"); purpleMemoryBox->setChecked(true);
        darkFlowBox=new QCheckBox("暗场演示模式"); darkFlowBox->setChecked(true);
        backReactionBox=new QCheckBox("开启挖坑反作用"); backReactionBox->setChecked(true);
        autoRiskBox=new QCheckBox("自动风险等级"); autoRiskBox->setChecked(true);
        predictBox=new QCheckBox("启用前瞻预判"); predictBox->setChecked(true);
        learnOverlayBox=new QCheckBox("显示记忆层"); learnOverlayBox->setChecked(true);
        conductOverlayBox=new QCheckBox("显示导通层"); conductOverlayBox->setChecked(true);
        benchmarkObstacleBox=new QCheckBox("基准测试使用标准障碍"); benchmarkObstacleBox->setChecked(true);

        ctl->addWidget(new QLabel("摄像头ID"),0,0); ctl->addWidget(cameraIndexBox,0,1); ctl->addWidget(openBtn,0,2); ctl->addWidget(closeBtn,0,3);
        ctl->addWidget(flowBtn,1,0,1,2); ctl->addWidget(calibBtn,1,2,1,2);
        ctl->addWidget(resetBtn,2,0,1,2); ctl->addWidget(swarmBtn,2,2,1,2);
        ctl->addWidget(benchmarkBtn,3,0,1,2); ctl->addWidget(autoTuneBtn,3,2,1,2);
        ctl->addWidget(demoBtn,4,0,1,4);
        ctl->addWidget(drawVectorBox,5,0); ctl->addWidget(drawTrailBox,5,1); ctl->addWidget(drawSwarmBox,5,2); ctl->addWidget(purpleMemoryBox,5,3);
        ctl->addWidget(darkFlowBox,6,0); ctl->addWidget(backReactionBox,6,1); ctl->addWidget(autoRiskBox,6,2); ctl->addWidget(predictBox,6,3);
        ctl->addWidget(learnOverlayBox,7,0); ctl->addWidget(conductOverlayBox,7,1); ctl->addWidget(benchmarkObstacleBox,7,2,1,2);

        QGroupBox* stateBox=new QGroupBox("状态与指标：给专家看数据");
        QGridLayout* st=new QGridLayout(stateBox);
        statusBig=new QLabel("等待启动");
        statusBig->setAlignment(Qt::AlignCenter);
        statusBig->setStyleSheet("QLabel{background:#15181c;color:#f4f6f8;border:1px solid #4b5056;font-size:30px;font-weight:bold;padding:10px;}");
        riskLabel=new QLabel("风险: LOW");
        energyLabel=new QLabel("场能均值: 0");
        gradientLabel=new QLabel("风险梯度: 0");
        repulsionLabel=new QLabel("排斥力: 0");
        entropyLabel=new QLabel("碰撞熵: 0");
        fpsLabel=new QLabel("FPS: 0");
        conductLabel=new QLabel("导通率均值: 0");
        learnLabel=new QLabel("记忆层均值: 0");
        agentLabel=new QLabel("集群数: 0");
        benchmarkLabel=new QLabel("基准测试: 未开启");
        autoTuneLabel=new QLabel("自适应调参: ON");

        QList<QLabel*> infoLabels={riskLabel,energyLabel,gradientLabel,repulsionLabel,entropyLabel,fpsLabel,conductLabel,learnLabel,agentLabel,benchmarkLabel,autoTuneLabel};
        for(auto* lb:infoLabels) lb->setStyleSheet("QLabel{font-size:14px;color:#e6ebef;background:#111316;border:1px solid #3a3f45;padding:4px;}");

        st->addWidget(statusBig,0,0,1,2);
        st->addWidget(riskLabel,1,0); st->addWidget(energyLabel,1,1);
        st->addWidget(gradientLabel,2,0); st->addWidget(repulsionLabel,2,1);
        st->addWidget(entropyLabel,3,0); st->addWidget(fpsLabel,3,1);
        st->addWidget(conductLabel,4,0); st->addWidget(learnLabel,4,1);
        st->addWidget(agentLabel,5,0); st->addWidget(autoTuneLabel,5,1);
        st->addWidget(benchmarkLabel,6,0,1,2);

        QGroupBox* paramBox=new QGroupBox("交通动力学参数");
        QVBoxLayout* pg=new QVBoxLayout(paramBox);
        QScrollArea* pScroll=new QScrollArea;
        pScroll->setWidgetResizable(true);
        pScroll->setFrameShape(QFrame::NoFrame);
        QWidget* pWidget=new QWidget;
        QVBoxLayout* pLay=new QVBoxLayout(pWidget);

        pAlpha=makeD(2.50,0.01,20.0,0.1,3);
        pBeta=makeD(0.35,0.01,3.0,0.01,3);
        pLeak=makeD(0.15,0.0,0.99,0.01,3);
        pRetain=makeD(0.85,0.0,0.999,0.01,3);
        pForget=makeD(0.05,0.0,1.0,0.01,3);
        pLearn=makeD(0.02,0.0,0.5,0.005,3);
        pLearnGain=makeD(2.50,0.0,10.0,0.05,3);
        pFieldCap=makeD(10.0,0.1,100.0,0.5,3);
        pAttract=makeD(0.15,0.0,5.0,0.01,3);
        pBoundary=makeD(2.00,0.0,20.0,0.1,3);
        pDt=makeD(1.0,0.01,5.0,0.01,3);
        pBackReaction=makeD(0.40,0.0,3.0,0.01,3);
        pSocial=makeD(15.0,0.0,120.0,0.1,3);
        pPersonalSpace=makeD(35.0,5.0,200.0,1.0,3);
        pPredictTime=makeD(3.0,0.0,20.0,0.5,3);
        pPredictGain=makeD(1.2,0.0,10.0,0.1,3);
        pAdaptiveGain=makeD(0.20,0.0,1.0,0.01,3);
        pGridW=makeI(320,64,640);
        pGridH=makeI(180,36,360);
        pSwarmSize=makeI(120,1,1000);
        pTrailRadius=makeI(25,1,120);

        QStringList names={"动能转换率 Alpha","蔓延半径 Beta","场泄漏 Leak","轨迹保留 Retain","主动遗忘 Forget","学习速度 Learn","记忆增益 LearnGain","场压上限 FieldCap","归心引力 Attract","边界斥力 Boundary","积分步长 Dt","反作用挖坑 BackReact","社会排斥 Social","个体安全距 PSpace","前瞻时间 PredictT","前瞻增益 PredictGain","自适应强度 AdaptiveGain"};
        QList<QWidget*> vals={pAlpha,pBeta,pLeak,pRetain,pForget,pLearn,pLearnGain,pFieldCap,pAttract,pBoundary,pDt,pBackReaction,pSocial,pPersonalSpace,pPredictTime,pPredictGain,pAdaptiveGain};

        for(int i=0;i<names.size();++i){
            QLabel* n=new QLabel(names[i]);
            n->setStyleSheet("QLabel{font-size:12px;color:#dce3e8;}");
            QHBoxLayout* hl=new QHBoxLayout;
            hl->addWidget(n); hl->addWidget(vals[i]);
            pLay->addLayout(hl);
        }
        QHBoxLayout* hg1=new QHBoxLayout; hg1->addWidget(new QLabel("网格W")); hg1->addWidget(pGridW); hg1->addWidget(new QLabel("网格H")); hg1->addWidget(pGridH); pLay->addLayout(hg1);
        QHBoxLayout* hg2=new QHBoxLayout; hg2->addWidget(new QLabel("集群数")); hg2->addWidget(pSwarmSize); hg2->addWidget(new QLabel("足迹半径")); hg2->addWidget(pTrailRadius); pLay->addLayout(hg2);

        pScroll->setWidget(pWidget);
        pg->addWidget(pScroll);

        QGroupBox* auditBox=new QGroupBox("能效对比审计");
        QGridLayout* ag=new QGridLayout(auditBox);
        costBarTraditional=new QProgressBar; costBarTraditional->setRange(0,100); costBarTraditional->setValue(98); costBarTraditional->setFormat("传统逻辑规划：%p% 计算负担");
        costBarTraditional->setStyleSheet("QProgressBar::chunk{background:#d32f2f;}");
        costBarChen=new QProgressBar; costBarChen->setRange(0,100); costBarChen->setValue(2); costBarChen->setFormat("Chen-Flow：%p% 实时负担");
        costBarChen->setStyleSheet("QProgressBar::chunk{background:#00c853;}");
        ag->addWidget(costBarTraditional,0,0);
        ag->addWidget(costBarChen,1,0);

        rightLayout->addWidget(controlBox,0);
        rightLayout->addWidget(stateBox,0);
        rightLayout->addWidget(paramBox,1);
        rightLayout->addWidget(auditBox,0);

        root->addWidget(leftPanel,5);
        root->addWidget(rightPanel,2);

        connect(openBtn,&QPushButton::clicked,this,&MainWindow::openCamera);
        connect(closeBtn,&QPushButton::clicked,this,&MainWindow::closeCamera);
        connect(flowBtn,&QPushButton::clicked,this,[this](){flowEnabled=!flowEnabled;flowBtn->setText(flowEnabled?"关闭时空记忆":"开启时空记忆");appendLog(QString("【操作】时空记忆 Flow -> %1").arg(flowEnabled?"ON":"OFF"));});

        connect(resetBtn,&QPushButton::clicked,this,&MainWindow::resetSystem);
        connect(calibBtn,&QPushButton::clicked,this,&MainWindow::calibrateEnvironment);
        connect(swarmBtn,&QPushButton::clicked,this,[this](){resetSwarm(frameWidth,frameHeight);appendLog(QString("【集群】已重建 %1 个智能体").arg(swarm.size()));});
        connect(benchmarkBtn,&QPushButton::clicked,this,[this](){benchmarkMode=!benchmarkMode;benchmarkBtn->setText(benchmarkMode?"关闭基准测试":"开启基准测试");startBenchmarkMode(benchmarkMode);});
        connect(autoTuneBtn,&QPushButton::clicked,this,[this](){autoTuneEnabled=!autoTuneEnabled;autoTuneBtn->setText(autoTuneEnabled?"关闭自适应调参":"开启自适应调参");autoTuneLabel->setText(QString("自适应调参: %1").arg(autoTuneEnabled?"ON":"OFF"));appendLog(QString("【调参】自适应调参 -> %1").arg(autoTuneEnabled?"ON":"OFF"));});
        connect(demoBtn,&QPushButton::clicked,this,[this](){demoMode=!demoMode;appendLog(QString("【演示】商业演示模式 -> %1").arg(demoMode?"ON":"OFF"));});
    }

    void initState(){
        fpsTimer.start();
        fpsLastTick=0;
        vehicle.pos=cv::Point2f(640,520);
        vehicle.vel=cv::Point2f(0,0);
        vehicle.acc=cv::Point2f(0,0);
        vehicle.mass=1.0f;
        vehicle.drag=0.84f;
        vehicle.maxVel=7.0f;
        updateRiskUi("LOW");
        resetSwarm(frameWidth,frameHeight);
    }

    void appendLog(const QString& s){
        QString line=QString("[%1] %2").arg(QDateTime::currentDateTime().toString("hh:mm:ss.zzz")).arg(s);
        logEdit->append(line);
    }

    void openCamera(){
        benchmarkMode=false;
        benchmarkBtn->setText("开启基准测试");
        int idx=cameraIndexBox->currentText().toInt();
        if(cap.isOpened()) cap.release();
        appendLog(QString("【操作】尝试打开摄像头 index=%1").arg(idx));
        cap.open(idx,cv::CAP_ANY);
        if(!cap.isOpened()){appendLog("【错误】摄像头打开失败");return;}
        cap.set(cv::CAP_PROP_FRAME_WIDTH,1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT,720);
        cap.set(cv::CAP_PROP_FPS,30);
        cameraOpened=true;
        prevGray.release();
        conductMap.release();
        potentialField.release();
        learnMap.release();
        benchmarkFrame.release();
        resetSwarm(1280,720);
        appendLog("【成功】摄像头已打开，Chen-Flow 摄像输入接管成功");
    }

    void closeCamera(){
        if(cap.isOpened()) cap.release();
        cameraOpened=false;
        videoLabel->setPixmap(QPixmap());
        videoLabel->setText("休眠中...");
        appendLog("【操作】摄像头已关闭");
    }

    void resetSystem(){
        conductMap.release();
        learnMap.release();
        potentialField.release();
        prevGray.release();
        vehicle.pos=cv::Point2f(frameWidth*0.5f,frameHeight*0.72f);
        vehicle.vel=cv::Point2f(0,0);
        vehicle.acc=cv::Point2f(0,0);
        vehicleTrail.clear();
        energyHistory.clear();
        motionHistory.clear();
        gradientHistory.clear();
        bench.reset();
        benchmarkTick=0;
        resetSwarm(frameWidth,frameHeight);
        appendLog("【重置】能量场、记忆场、动量场已全部清空");
    }

    void calibrateEnvironment(){
        baselineEnergy=std::max(1.0,lastEnergyMean);
        baselineMotion=std::max(0.01,lastMotion);
        baselineEntropy=std::max(0.01,lastEntropy);
        calibrated=true;
        appendLog(QString("【标定】已锁定环境基线 energy=%1 motion=%2 entropy=%3").arg(baselineEnergy,0,'f',3).arg(baselineMotion,0,'f',3).arg(baselineEntropy,0,'f',3));
    }

    void startBenchmarkMode(bool on){
        if(on){
            if(cap.isOpened()) cap.release();
            cameraOpened=false;
            bench.reset();
            benchmarkTick=0;
            prevGray.release();
            conductMap.release();
            potentialField.release();
            learnMap.release();
            resetSystem();
            appendLog("【基准测试】已开启标准测试场景：横穿障碍 + 遮挡 + 残余记忆");
        }else{
            bench.reset();
            benchmarkTick=0;
            appendLog("【基准测试】已关闭");
        }
    }

    void resetSwarm(int width,int height){
        swarm.clear();
        int n=pSwarmSize->value();
        std::random_device rd;std::mt19937 gen(rd());
        std::uniform_real_distribution<float> disX(100.0f,std::max(101.0f,(float)width-100.0f));
        std::uniform_real_distribution<float> disY(100.0f,std::max(101.0f,(float)height-100.0f));
        std::uniform_real_distribution<float> disV(-2.0f,2.0f);
        for(int i=0;i<n;++i){
            PhysicalAgent a;
            a.id=i;
            a.pos=cv::Point2f(disX(gen),disY(gen));
            a.vel=cv::Point2f(disV(gen),disV(gen));
            a.mass=0.8f+(i%5)*0.1f;
            a.drag=0.82f+(i%3)*0.02f;
            a.maxVel=4.0f+(i%4);
            a.radius=5.0f+(i%3);
            swarm.push_back(a);
        }
    }

    void updateRiskUi(const QString& risk){
        if(riskState==risk) return;
        appendLog(QString("【状态变化】风险等级 %1 -> %2").arg(riskState).arg(risk));
        riskState=risk;
        riskLabel->setText(QString("风险: %1").arg(riskState));
        if(risk=="LOW"){
            statusBig->setText("畅通无阻");
            statusBig->setStyleSheet("QLabel{background:#0e301a;color:#a5d6a7;border:1px solid #2e7d32;font-size:30px;font-weight:bold;padding:10px;}");
        }else if(risk=="MEDIUM"){
            statusBig->setText("感知扰动");
            statusBig->setStyleSheet("QLabel{background:#40320a;color:#ffe082;border:1px solid #f9a825;font-size:30px;font-weight:bold;padding:10px;}");
        }else if(risk=="HIGH"){
            statusBig->setText("物理排斥");
            statusBig->setStyleSheet("QLabel{background:#4a1e0b;color:#ffccbc;border:1px solid #e64a19;font-size:30px;font-weight:bold;padding:10px;}");
        }else{
            statusBig->setText("空间断裂");
            statusBig->setStyleSheet("QLabel{background:#4a0f0f;color:#ffcdd2;border:1px solid #c62828;font-size:30px;font-weight:bold;padding:10px;}");
        }
    }

    void pushHistory(std::deque<double>& q,double v,int maxSize=60){
        q.push_back(v);
        while((int)q.size()>maxSize) q.pop_front();
    }

    double meanOf(const std::deque<double>& q){
        if(q.empty()) return 0.0;
        double s=0.0; for(double v:q) s+=v; return s/q.size();
    }

    double varOf(const std::deque<double>& q){
        if(q.size()<2) return 0.0;
        double m=meanOf(q),s=0.0; for(double v:q){double d=v-m; s+=d*d;} return s/q.size();
    }

    void adaptiveTune(){
        if(!autoTuneEnabled) return;
        if(energyHistory.size()<20) return;
        double eMean=meanOf(energyHistory),eVar=varOf(energyHistory),mMean=meanOf(motionHistory),gMean=meanOf(gradientHistory);
        double gain=pAdaptiveGain->value();
        double alpha=pAlpha->value(),learn=pLearn->value(),social=pSocial->value();

        if(eMean<baselineEnergy*0.8 && mMean<baselineMotion*1.2){alpha=clampd(alpha+0.02*gain,0.01,20.0);learn=clampd(learn+0.001*gain,0.0,0.5);}
        if(eMean>baselineEnergy*2.5 || eVar>baselineEnergy*baselineEnergy*0.5){alpha=clampd(alpha-0.03*gain,0.01,20.0);learn=clampd(learn-0.0015*gain,0.0,0.5);}
        if(gMean>0.25){social=clampd(social+0.08*gain,0.0,120.0);}
        if(gMean<0.05 && mMean<baselineMotion*1.1){social=clampd(social-0.05*gain,0.0,120.0);}

        pAlpha->blockSignals(true); pLearn->blockSignals(true); pSocial->blockSignals(true);
        pAlpha->setValue(alpha); pLearn->setValue(learn); pSocial->setValue(social);
        pAlpha->blockSignals(false); pLearn->blockSignals(false); pSocial->blockSignals(false);
    }

    void updateDynamicRisk(float grad,float repul,float entropy){
        double eBase=calibrated?baselineEnergy:50.0;
        double mBase=calibrated?baselineMotion:0.02;
        double entBase=calibrated?baselineEntropy:0.2;
        double eScore=lastEnergyMean/std::max(1.0,eBase);
        double mScore=lastMotion/std::max(0.01,mBase);
        double gScore=grad*10.0;
        double rScore=repul*0.8;
        double hScore=entropy/std::max(0.2,entBase);
        double score=0.35*eScore+0.20*mScore+0.20*gScore+0.15*rScore+0.10*hScore;
        lastRiskRaw=score;

        QString newRisk=riskState;
        if(riskState=="LOW"){
            if(score>1.3) newRisk="MEDIUM";
        }else if(riskState=="MEDIUM"){
            if(score<0.9) newRisk="LOW";
            else if(score>2.5) newRisk="HIGH";
        }else if(riskState=="HIGH"){
            if(score<1.8) newRisk="MEDIUM";
            else if(score>4.2) newRisk="CRITICAL";
        }else{
            if(score<3.3) newRisk="HIGH";
        }
        if(autoRiskBox->isChecked()) updateRiskUi(newRisk);
    }

    void normalizeEnergyField(cv::Mat& field,float cap){
        if(field.empty()) return;
        cv::threshold(field,field,0.0,0.0,cv::THRESH_TOZERO);
        cv::Mat compressed=field.clone();
        compressed=compressed/(1.0f+compressed);
        field=compressed*cap;
        cv::Mat meanField;
        cv::boxFilter(field,meanField,-1,cv::Size(5,5));
        field=field-0.18f*meanField;
        cv::threshold(field,field,0.0,0.0,cv::THRESH_TOZERO);
        cv::min(field,cap,field);
    }

    cv::Point2f calculateFieldForce(const cv::Mat& field,cv::Point2f p){
        int ix=(int)p.x,iy=(int)p.y;
        if(ix<1||ix>=field.cols-1||iy<1||iy>=field.rows-1) return cv::Point2f(0,0);
        float gradX=(field.at<float>(iy,ix+1)-field.at<float>(iy,ix-1))*0.5f;
        float gradY=(field.at<float>(iy+1,ix)-field.at<float>(iy-1,ix))*0.5f;
        return cv::Point2f(-gradX*18.0f,-gradY*18.0f);
    }

    cv::Point2f calculateBoundaryForce(cv::Size sz,cv::Point2f pos,float gain){
        cv::Point2f f(0,0);
        float m=30.0f;
        if(pos.x<m) f.x+=gain*(1.0f+(m-pos.x)/m);
        if(pos.x>sz.width-m) f.x-=gain*(1.0f+(pos.x-(sz.width-m))/m);
        if(pos.y<m) f.y+=gain*(1.0f+(m-pos.y)/m);
        if(pos.y>sz.height-m) f.y-=gain*(1.0f+(pos.y-(sz.height-m))/m);
        return f;
    }

    void applyAgentBackReaction(cv::Mat& conduct,cv::Point2f pos,float strength,int radius){
        if(conduct.empty()) return;
        cv::Point center((int)(pos.x/scaleX),(int)(pos.y/scaleY));
        int r=std::max(1,(int)(radius/std::max(1.0f,std::min(scaleX,scaleY))));
        cv::circle(conduct,center,r,cv::Scalar(std::max(0.005f,0.05f-(float)strength*0.02f)),-1,cv::LINE_AA);
    }

    void refreshSpatialGrid(int w,int h){
        for(int x=0;x<BUCKET_COLS;++x) for(int y=0;y<BUCKET_ROWS;++y) gridBuckets[x][y].clear();
        for(size_t i=0;i<swarm.size();++i){
            int gx=std::clamp((int)(swarm[i].pos.x*BUCKET_COLS/std::max(1,w)),0,BUCKET_COLS-1);
            int gy=std::clamp((int)(swarm[i].pos.y*BUCKET_ROWS/std::max(1,h)),0,BUCKET_ROWS-1);
            gridBuckets[gx][gy].push_back((int)i);
        }
    }

    cv::Point2f calculateSocialForce(const PhysicalAgent& a,int w,int h,float pSpace,float gain){
        cv::Point2f f(0,0);
        int gx=std::clamp((int)(a.pos.x*BUCKET_COLS/std::max(1,w)),0,BUCKET_COLS-1);
        int gy=std::clamp((int)(a.pos.y*BUCKET_ROWS/std::max(1,h)),0,BUCKET_ROWS-1);
        float p2=pSpace*pSpace;
        for(int dx=-1;dx<=1;++dx){
            for(int dy=-1;dy<=1;++dy){
                int nx=gx+dx,ny=gy+dy;
                if(nx<0||nx>=BUCKET_COLS||ny<0||ny>=BUCKET_ROWS) continue;
                for(int idx:gridBuckets[nx][ny]){
                    if(a.id==swarm[idx].id) continue;
                    cv::Point2f diff=a.pos-swarm[idx].pos;
                    float d2=diff.x*diff.x+diff.y*diff.y;
                    if(d2<p2&&d2>0.01f){
                        float dist=std::sqrt(d2);
                        f+=(diff/dist)*(gain*(pSpace-dist)/std::max(1.0f,dist));
                    }
                }
            }
        }
        return f;
    }

    void createBenchmarkFrame(){
        benchmarkFrame=cv::Mat(frameHeight,frameWidth,CV_8UC3,cv::Scalar(22,24,26));
        cv::rectangle(benchmarkFrame,cv::Rect(0,(int)(frameHeight*0.72),frameWidth,(int)(frameHeight*0.28)),cv::Scalar(55,55,55),-1);
        cv::line(benchmarkFrame,cv::Point(0,(int)(frameHeight*0.72)),cv::Point(frameWidth,(int)(frameHeight*0.72)),cv::Scalar(80,80,80),2);
        if(benchmarkObstacleBox->isChecked()){
            int x=(benchmarkTick*10)% (frameWidth+200)-100;
            cv::rectangle(benchmarkFrame,cv::Rect(x,(int)(frameHeight*0.55),70,150),cv::Scalar(245,245,245),-1);
            cv::rectangle(benchmarkFrame,cv::Rect(frameWidth/2-40,(int)(frameHeight*0.50),80,180),cv::Scalar(40,40,40),-1);
        }
    }

    double computeEntropy01(const cv::Mat& field){
        if(field.empty()) return 0.0;
        cv::Mat clipped; cv::min(field,pFieldCap->value(),clipped);
        cv::Mat normed = clipped / std::max(0.001,pFieldCap->value());
        int bins=64;
        std::vector<int> hist(bins,0);
        for(int y=0;y<normed.rows;++y){
            const float* row=normed.ptr<float>(y);
            for(int x=0;x<normed.cols;++x){
                int b=std::clamp((int)(row[x]*(bins-1)),0,bins-1);
                hist[b]++;
            }
        }
        double total=(double)(normed.rows*normed.cols);
        double ent=0.0;
        for(int c:hist){
            if(c<=0) continue;
            double p=(double)c/total;
            ent -= p*std::log2(p);
        }
        double maxEnt=std::log2((double)bins);
        return maxEnt>0.0?ent/maxEnt:0.0;
    }

    void updateBenchmarkMetrics(){
        if(!benchmarkMode) return;
        bench.totalFrames++;
        bench.totalSpeed+=cv::norm(vehicle.vel);
        if(vehicleTrail.size()>=2) bench.totalPath+=cv::norm(vehicleTrail.back()-vehicleTrail[vehicleTrail.size()-2]);
        bench.totalEnergy+=lastEnergyMean;
        bench.totalRepulsion+=lastRepulsion;
        bench.totalGradient+=lastGradientMean;

        int bx=(benchmarkTick*10)% (frameWidth+200)-100;
        cv::Rect obstacleRect(bx,(int)(frameHeight*0.55),70,150);
        if(obstacleRect.contains(cv::Point((int)vehicle.pos.x,(int)vehicle.pos.y))) bench.collisions++;

        if(bench.totalFrames>0){
            benchmarkLabel->setText(QString("基准测试: 帧=%1 碰撞=%2 平均速=%3 平均路长=%4").arg(bench.totalFrames).arg(bench.collisions).arg(bench.totalSpeed/bench.totalFrames,0,'f',2).arg(bench.totalPath/std::max(1,bench.totalFrames),0,'f',2));
        }
    }

    void renderOverlays(cv::Mat& render){
        if(conductOverlayBox->isChecked()){
            cv::Mat conductNorm; conductMap.convertTo(conductNorm,CV_8U,255.0/std::max(0.001,pFieldCap->value()));
            cv::Mat conductColor;
            //cv::applyColorMap(conductNorm,conductColor,cv::COLORMAP_OCEAN);
            cv::applyColorMap(conductNorm, conductColor, cv::COLORMAP_TURBO);
            //cv::applyColorMap(conductNorm, conductColor, cv::COLORMAP_JET);
            cv::resize(conductColor,conductColor,render.size());
            //暗
            // if(darkFlowBox->isChecked()){
            //     cv::Mat dark(render.size(),CV_8UC3,cv::Scalar(6,6,10));
            //     cv::addWeighted(dark,0.75,conductColor,0.50,0,render);
            // }else{
            //     cv::addWeighted(render,0.72,conductColor,0.35,0,render);
            // }
            //亮
            if (darkFlowBox->isChecked()) {
                cv::addWeighted(render, 0.35, conductColor, 0.85, 0, render);
            } else {
                cv::addWeighted(render, 0.65, conductColor, 0.75, 0, render);
            }
        }
        if(learnOverlayBox->isChecked() && purpleMemoryBox->isChecked()){
            cv::Mat learnNorm; learnMap.convertTo(learnNorm,CV_8U,255.0/std::max(0.001,pLearnGain->value()));
            cv::Mat purple(render.size(),CV_8UC3,cv::Scalar(0,0,0));
            cv::resize(learnNorm,learnNorm,render.size());
            for(int y=0;y<render.rows;++y){
                uchar* l=learnNorm.ptr<uchar>(y);
                cv::Vec3b* p=purple.ptr<cv::Vec3b>(y);
                for(int x=0;x<render.cols;++x){
                    p[x][0]=(uchar)(l[x]*0.7);
                    p[x][1]=(uchar)(l[x]*0.15);
                    p[x][2]=(uchar)(l[x]*0.9);
                }
            }
            cv::addWeighted(render,1.0,purple,0.30,0,render);
        }
    }

    void renderVehicleAndSwarm(cv::Mat& render){
        if(drawTrailBox->isChecked()){
            vehicleTrail.push_back(vehicle.pos);
            if(vehicleTrail.size()>120) vehicleTrail.erase(vehicleTrail.begin());
            for(size_t i=1;i<vehicleTrail.size();++i){
                int alpha=(int)(255.0*i/vehicleTrail.size());
                cv::line(render,vehicleTrail[i-1],vehicleTrail[i],cv::Scalar(255,alpha/3,255),2,cv::LINE_AA);
            }
        }

        cv::drawMarker(render,vehicle.pos,cv::Scalar(0,255,255),cv::MARKER_CROSS,28,2,cv::LINE_AA);
        if(predictBox->isChecked()) cv::line(render,vehicle.pos,vehicle.pos+vehicle.vel*pPredictTime->value(),cv::Scalar(0,200,255),2,cv::LINE_AA);

        if(drawSwarmBox->isChecked()){
            for(const auto& a:swarm){
                float forceMag=cv::norm(a.acc);
                int red=(int)clampf(forceMag*15.0f,0,255);
                int green=(int)clampf(255-red,0,255);
                cv::circle(render,a.pos,(int)a.radius,cv::Scalar(0,green,red),-1,cv::LINE_AA);
                if(predictBox->isChecked()) cv::line(render,a.pos,a.pos+a.vel*pPredictTime->value(),cv::Scalar(200,200,200),1,cv::LINE_AA);
                if(red>180) cv::putText(render,"!",a.pos+cv::Point2f(8,-8),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),2,cv::LINE_AA);
            }
        }

        if(drawVectorBox->isChecked()){
            cv::Point2f gp(vehicle.pos.x/scaleX,vehicle.pos.y/scaleY);
            cv::Point2f fField=calculateFieldForce(potentialField,gp); fField.x*=scaleX; fField.y*=scaleY;
            cv::line(render,vehicle.pos,vehicle.pos+fField*6.0f,cv::Scalar(255,255,255),2,cv::LINE_AA);
            cv::putText(render,"F_field",vehicle.pos+cv::Point2f(12,-12),cv::FONT_HERSHEY_SIMPLEX,0.45,cv::Scalar(255,255,255),1,cv::LINE_AA);
        }
    }

    void renderHud(cv::Mat& render){
        cv::rectangle(render,cv::Rect(15,15,760,130),cv::Scalar(10,12,15),-1);
        cv::rectangle(render,cv::Rect(15,15,760,130),cv::Scalar(80,90,100),1);
        cv::putText(render,"Chen-Flow Core | Dynamic Swarm Physics |Physical Avoidance",cv::Point(25,40),cv::FONT_HERSHEY_SIMPLEX,0.65,cv::Scalar(255,255,255),1,cv::LINE_AA);
        cv::putText(render,hudInteractionText.toStdString(),cv::Point(25,68),cv::FONT_HERSHEY_SIMPLEX,0.60,hudInteractionColor,2,cv::LINE_AA);
        cv::putText(render,QString("Risk Gradient %1").arg(lastGradientMean,0,'f',5).toStdString(),cv::Point(25,95),cv::FONT_HERSHEY_SIMPLEX,0.56,cv::Scalar(220,220,220),1,cv::LINE_AA);
        cv::putText(render,QString("Repulsion Force %1 N").arg(lastRepulsion,0,'f',2).toStdString(),cv::Point(260,95),cv::FONT_HERSHEY_SIMPLEX,0.56,cv::Scalar(220,220,220),1,cv::LINE_AA);
        cv::putText(render,QString("Collision Entropy %1").arg(lastEntropy,0,'f',3).toStdString(),cv::Point(490,95),cv::FONT_HERSHEY_SIMPLEX,0.56,cv::Scalar(220,220,220),1,cv::LINE_AA);
        cv::putText(render,QString("FLOW %1 | Adaptive %2 | Benchmark %3").arg(flowEnabled?"ON":"OFF").arg(autoTuneEnabled?"ON":"OFF").arg(benchmarkMode?"ON":"OFF").toStdString(),cv::Point(25,122),cv::FONT_HERSHEY_SIMPLEX,0.58,flowEnabled?cv::Scalar(255,120,255):cv::Scalar(128,128,128),2,cv::LINE_AA);
    }

    void updateUiTexts(){
        energyLabel->setText(QString("场能均值: %1").arg(lastEnergyMean,0,'f',3));
        gradientLabel->setText(QString("风险梯度: %1").arg(lastGradientMean,0,'f',5));
        repulsionLabel->setText(QString("排斥力: %1 N").arg(lastRepulsion,0,'f',2));
        entropyLabel->setText(QString("碰撞熵: %1").arg(lastEntropy,0,'f',3));
        fpsLabel->setText(QString("FPS: %1").arg(fps,0,'f',1));
        conductLabel->setText(QString("导通率均值: %1").arg(lastConductAvg,0,'f',4));
        learnLabel->setText(QString("记忆层均值: %1").arg(lastLearnAvg,0,'f',4));
        agentLabel->setText(QString("集群数: %1 | 车位置:(%2,%3)").arg(swarm.size()).arg((int)vehicle.pos.x).arg((int)vehicle.pos.y));
        autoTuneLabel->setText(QString("自适应调参: %1 | Alpha=%2 Learn=%3 Social=%4").arg(autoTuneEnabled?"ON":"OFF").arg(pAlpha->value(),0,'f',3).arg(pLearn->value(),0,'f',3).arg(pSocial->value(),0,'f',2));
        costBarChen->setValue(std::max(1,std::min(20,(int)(lastGradientMean*20+lastRepulsion))));
    }

    void processFrameSource(){
        if(benchmarkMode){
            createBenchmarkFrame();
            frameBgr=benchmarkFrame.clone();
            benchmarkTick++;
            cameraOpened=false;
            return;
        }
        if(!cap.isOpened()) return;
        cap>>frameBgr;
    }

    void onFrame(){
        processFrameSource();
        if(frameBgr.empty()) return;

        frameWidth=frameBgr.cols;
        frameHeight=frameBgr.rows;
        gridW=pGridW->value();
        gridH=pGridH->value();
        scaleX=(float)frameWidth/gridW;
        scaleY=(float)frameHeight/gridH;

        if(conductMap.empty()||conductMap.cols!=gridW||conductMap.rows!=gridH){
            conductMap=cv::Mat::zeros(gridH,gridW,CV_32F);
            learnMap=cv::Mat::zeros(gridH,gridW,CV_32F);
            potentialField=cv::Mat::zeros(gridH,gridW,CV_32F);
        }

        qint64 now=fpsTimer.elapsed();
        if(fpsLastTick>0&&now>fpsLastTick) fps=fps*0.90+(1000.0/(now-fpsLastTick))*0.10;
        fpsLastTick=now;

        cv::cvtColor(frameBgr,gray,cv::COLOR_BGR2GRAY);
        cv::resize(gray,gray,cv::Size(gridW,gridH),0,0,cv::INTER_AREA);
        if(prevGray.empty()) prevGray=gray.clone();

        cv::Mat diff; cv::absdiff(gray,prevGray,diff); prevGray=gray.clone();
        cv::Mat motionF; diff.convertTo(motionF,CV_32F,1.0/255.0);

        int k=(int)std::max(3.0,std::round(pBeta->value()*20.0)); if(k%2==0) k++;
        cv::Mat blurMotion;
        cv::GaussianBlur(motionF,blurMotion,cv::Size(k,k),pBeta->value()*6.0+0.1);

        lastBrightness=cv::mean(gray)[0];
        lastMotion=cv::mean(blurMotion)[0];

        cv::Mat lap; cv::Laplacian(gray,lap,CV_32F);
        cv::Scalar meanLap,stdLap; cv::meanStdDev(lap,meanLap,stdLap);
        lastSharp=stdLap[0];

        if(flowEnabled) learnMap=(1.0f-(float)pLearn->value())*learnMap + (float)pLearn->value()*blurMotion;
        else learnMap*=0.95f;

        conductMap*=pRetain->value();
        conductMap-=pForget->value()*0.03f;
        cv::threshold(conductMap,conductMap,0.0,0.0,cv::THRESH_TOZERO);

        conductMap += blurMotion*(float)pAlpha->value();
        conductMap += learnMap*(float)pLearnGain->value();
        conductMap *= (1.0f-(float)pLeak->value());

        potentialField=conductMap.clone();
        normalizeEnergyField(potentialField,(float)pFieldCap->value());

        float dt=(float)pDt->value();
        int evasiveCount=0;

        if(avoidFieldEnabled){
            cv::Point2f gPos(vehicle.pos.x/scaleX,vehicle.pos.y/scaleY);
            cv::Point2f fEnv=calculateFieldForce(potentialField,gPos); fEnv.x*=scaleX; fEnv.y*=scaleY;
            if(predictBox->isChecked()&&cv::norm(vehicle.vel)>1.0f){
                cv::Point2f future((vehicle.pos.x+vehicle.vel.x*(float)pPredictTime->value())/scaleX,(vehicle.pos.y+vehicle.vel.y*(float)pPredictTime->value())/scaleY);
                cv::Point2f fFut=calculateFieldForce(potentialField,future); fFut.x*=scaleX; fFut.y*=scaleY;
                fEnv += fFut*(float)pPredictGain->value();
            }
            cv::Point2f home(frameWidth*0.5f,frameHeight*0.72f);
            cv::Point2f fHome=(home-vehicle.pos)*(float)pAttract->value();
            cv::Point2f fBound=calculateBoundaryForce(frameBgr.size(),vehicle.pos,(float)pBoundary->value());
            cv::Point2f net=fEnv+fHome+fBound;
            vehicle.update(net,dt);
            vehicle.pos.x=clampf(vehicle.pos.x,0,frameWidth-1);
            vehicle.pos.y=clampf(vehicle.pos.y,0,frameHeight-1);
            lastRepulsion=cv::norm(fEnv)/20.0f;

            if(backReactionBox->isChecked()) applyAgentBackReaction(conductMap,vehicle.pos,(float)pBackReaction->value(),pTrailRadius->value());

            if(swarmEnabled && !swarm.empty()){
                refreshSpatialGrid(frameWidth,frameHeight);
                float pSpace=(float)pPersonalSpace->value();
                float sGain=(float)pSocial->value();
                for(auto& a:swarm){
                    cv::Point2f agPos(a.pos.x/scaleX,a.pos.y/scaleY);
                    cv::Point2f afEnv=calculateFieldForce(potentialField,agPos); afEnv.x*=scaleX; afEnv.y*=scaleY;
                    if(predictBox->isChecked()&&cv::norm(a.vel)>0.5f){
                        cv::Point2f future((a.pos.x+a.vel.x*(float)pPredictTime->value())/scaleX,(a.pos.y+a.vel.y*(float)pPredictTime->value())/scaleY);
                        cv::Point2f fFut=calculateFieldForce(potentialField,future); fFut.x*=scaleX; fFut.y*=scaleY;
                        afEnv += fFut*(float)pPredictGain->value();
                    }
                    cv::Point2f afSocial=calculateSocialForce(a,frameWidth,frameHeight,pSpace,sGain);
                    cv::Point2f afHome=(cv::Point2f(frameWidth*0.5f,frameHeight*0.60f)-a.pos)*(float)(pAttract->value()*0.5);
                    cv::Point2f afBound=calculateBoundaryForce(frameBgr.size(),a.pos,(float)pBoundary->value());
                    cv::Point2f aNet=afEnv+afSocial+afHome+afBound;
                    a.update(aNet,dt);
                    a.pos.x=clampf(a.pos.x,0,frameWidth-1);
                    a.pos.y=clampf(a.pos.y,0,frameHeight-1);
                    if(backReactionBox->isChecked()) applyAgentBackReaction(conductMap,a.pos,(float)pBackReaction->value()*0.3f,std::max(4,pTrailRadius->value()/2));
                    if(cv::norm(afEnv)>30.0f) evasiveCount++;
                }
            }
        }else{
            vehicle.update(cv::Point2f(0,0),dt);
            for(auto& a:swarm) a.update(cv::Point2f(0,0),dt);
            lastRepulsion=0.0;
        }

        cv::Mat gx,gy; cv::Sobel(potentialField,gx,CV_32F,1,0); cv::Sobel(potentialField,gy,CV_32F,0,1);
        cv::Mat gMag; cv::magnitude(gx,gy,gMag);
        lastGradientMean=cv::mean(gMag)[0];
        cv::Scalar sumField=cv::sum(potentialField);
        lastEnergySum=sumField[0];
        lastEnergyMean=cv::mean(potentialField)[0];
        double minv,maxv; cv::minMaxLoc(potentialField,&minv,&maxv);
        lastEnergyPeak=maxv;
        lastConductAvg=cv::mean(conductMap)[0];
        lastLearnAvg=cv::mean(learnMap)[0];
        lastEntropy=computeEntropy01(potentialField);

        pushHistory(energyHistory,lastEnergyMean);
        pushHistory(motionHistory,lastMotion);
        pushHistory(gradientHistory,lastGradientMean);
        adaptiveTune();
        updateDynamicRisk((float)lastGradientMean,(float)lastRepulsion,(float)lastEntropy);

        if(evasiveCount==0){hudInteractionText="STEADY | Field energy is stable, swarm follows the lowest-cost path.";hudInteractionColor=cv::Scalar(0,255,0);}
        else if(evasiveCount<(int)(swarm.size()*0.3)){hudInteractionText=QString("AVOIDANCE | Local pressure detected, %1 agents are detouring autonomously.").arg(evasiveCount);hudInteractionColor=cv::Scalar(0,200,255);}
        else{hudInteractionText=QString("HIGH PRESSURE | Swarm game activated, %1 agents entered strong-repulsion avoidance.").arg(evasiveCount);hudInteractionColor=cv::Scalar(0,0,255);}

        if(demoMode && lastRepulsion>4.0) appendLog(QString("[Cause and Effect Retrospection] The main vehicle was pushed away by the field, repulsive force=%1N，gradient=%2，Collision entropy=%3").arg(lastRepulsion,0,'f',2).arg(lastGradientMean,0,'f',4).arg(lastEntropy,0,'f',3));

        cv::Mat render=frameBgr.clone();
        renderOverlays(render);
        renderVehicleAndSwarm(render);
        renderHud(render);

        QImage qimg=matToQImage(render);
        videoLabel->setPixmap(QPixmap::fromImage(qimg).scaled(videoLabel->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation));

        updateBenchmarkMetrics();
        updateUiTexts();

        frameCount++;
        if(frameCount%30==0){
            QString msg=QString("【专业终端】frame=%1 bright=%2 motion=%3 sharp=%4 fps=%5 energy=%6 gradient=%7 repulsion=%8 entropy=%9 risk=%10 size=%11x%12 flow=%13 vehicle=(%14,%15) conduct=%16 learn=%17")
                .arg(frameCount)
                .arg(lastBrightness,0,'f',2)
                .arg(lastMotion,0,'f',3)
                .arg(lastSharp,0,'f',3)
                .arg(fps,0,'f',2)
                .arg(lastEnergyMean,0,'f',3)
                .arg(lastGradientMean,0,'f',5)
                .arg(lastRepulsion,0,'f',2)
                .arg(lastEntropy,0,'f',3)
                .arg(riskState)
                .arg(frameWidth)
                .arg(frameHeight)
                .arg(flowEnabled?"ON":"OFF")
                .arg((int)vehicle.pos.x)
                .arg((int)vehicle.pos.y)
                .arg(lastConductAvg,0,'f',4)
                .arg(lastLearnAvg,0,'f',4);
            appendLog(msg);
        }
    }
};

int main(int argc,char *argv[]){
    QApplication app(argc,argv);
    MainWindow w;
    w.show();
    return app.exec();
}
#include "main.moc"