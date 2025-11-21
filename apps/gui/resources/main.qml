import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    visible: true
    width: 1200
    height: 900
    title: "T2D-VAT Suite"

    FileDialog {
        id: fileDialog
        title: "Select Input Excel File"
        nameFilters: ["Excel files (*.xlsx *.xls)", "All files (*)"]
        onAccepted: {
            viewModel.inputPath = selectedFile
        }
    }

    FolderDialog {
        id: folderDialog
        title: "Select Output Directory"
        onAccepted: {
            viewModel.outputDir = selectedFolder
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        // Header Section
        GroupBox {
            title: "Configuration"
            Layout.fillWidth: true
            Layout.margins: 10

            GridLayout {
                columns: 3
                rowSpacing: 10
                columnSpacing: 10
                width: parent.width

                Label { text: "Input File:" }
                TextField {
                    text: viewModel.inputPath
                    Layout.fillWidth: true
                    readOnly: true
                    placeholderText: "Select input Excel file..."
                }
                Button {
                    text: "Browse..."
                    onClicked: fileDialog.open()
                    enabled: !viewModel.isBusy
                }

                Label { text: "Output Dir:" }
                TextField {
                    text: viewModel.outputDir
                    Layout.fillWidth: true
                    readOnly: true
                    placeholderText: "Select output directory..."
                }
                Button {
                    text: "Browse..."
                    onClicked: folderDialog.open()
                    enabled: !viewModel.isBusy
                }
            }
        }

        TabBar {
            id: bar
            width: parent.width
            TabButton { text: "Reproducer" }
            TabButton { text: "Injector (ML)" }
        }

        StackLayout {
            width: parent.width
            currentIndex: bar.currentIndex
            Layout.fillHeight: true
            Layout.margins: 10

            // Reproducer Tab
            Item {
                ColumnLayout {
                    anchors.fill: parent
                    
                    RowLayout {
                        Label { text: "Top N Proteins:" }
                        SpinBox {
                            from: 1
                            to: 100
                            value: viewModel.reproducerTopN
                            onValueModified: viewModel.reproducerTopN = value
                            editable: true
                            enabled: !viewModel.isBusy
                        }
                        Button {
                            text: "Run Reproducer"
                            highlighted: true
                            enabled: !viewModel.isBusy && viewModel.inputPath !== "" && viewModel.outputDir !== ""
                            onClicked: viewModel.runReproducer()
                        }
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true

                        Flow {
                            width: parent.width
                            spacing: 10
                            
                            Repeater {
                                model: viewModel.reproducerImages
                                delegate: Column {
                                    width: 550
                                    height: 450
                                    spacing: 5
                                    
                                    Image {
                                        source: modelData
                                        width: 550
                                        height: 400
                                        fillMode: Image.PreserveAspectFit
                                        cache: false 
                                        sourceSize.width: 550
                                        sourceSize.height: 400
                                        
                                        Rectangle {
                                            anchors.fill: parent
                                            color: "transparent"
                                            border.color: "#ccc"
                                        }
                                    }
                                    Text {
                                        text: modelData.split("/").pop()
                                        width: parent.width
                                        horizontalAlignment: Text.AlignHCenter
                                        elide: Text.ElideMiddle
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Injector Tab
            Item {
                ColumnLayout {
                    anchors.fill: parent
                    
                    RowLayout {
                        Label { text: "Top Features:" }
                        SpinBox {
                            from: 1
                            to: 100
                            value: viewModel.injectorTopN
                            onValueModified: viewModel.injectorTopN = value
                            editable: true
                            enabled: !viewModel.isBusy
                        }
                        Button {
                            text: "Run Injector"
                            highlighted: true
                            enabled: !viewModel.isBusy && viewModel.inputPath !== "" && viewModel.outputDir !== ""
                            onClicked: viewModel.runInjector()
                        }
                    }

                    SplitView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        orientation: Qt.Horizontal

                        // Metrics Area
                        GroupBox {
                            title: "Metrics"
                            SplitView.preferredWidth: 300
                            SplitView.minimumWidth: 200
                            
                            ScrollView {
                                anchors.fill: parent
                                TextArea {
                                    text: viewModel.injectorMetrics
                                    readOnly: true
                                    font.family: "Courier"
                                }
                            }
                        }

                        // Plots Area
                        ScrollView {
                            SplitView.fillWidth: true
                            clip: true

                            Flow {
                                width: parent.width
                                spacing: 10
                                
                                Repeater {
                                    model: viewModel.injectorImages
                                    delegate: Column {
                                        width: 400
                                        height: 350
                                        spacing: 5
                                        
                                        Image {
                                            source: modelData
                                            width: 400
                                            height: 300
                                            fillMode: Image.PreserveAspectFit
                                            cache: false
                                            sourceSize.width: 400
                                            sourceSize.height: 300

                                            Rectangle {
                                                anchors.fill: parent
                                                color: "transparent"
                                                border.color: "#ccc"
                                            }
                                        }
                                        Text {
                                            text: modelData.split("/").pop()
                                            width: parent.width
                                            horizontalAlignment: Text.AlignHCenter
                                            elide: Text.ElideMiddle
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Status Bar
        Rectangle {
            Layout.fillWidth: true
            height: 30
            color: "#eee"
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 5
                
                Label {
                    text: viewModel.statusMessage
                    Layout.fillWidth: true
                    elide: Text.ElideRight
                }
                
                BusyIndicator {
                    running: viewModel.isBusy
                    Layout.preferredHeight: 20
                    Layout.preferredWidth: 20
                    visible: viewModel.isBusy
                }
            }
        }
    }
}
